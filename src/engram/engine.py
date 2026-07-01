"""記憶エンジン(担当: Agent B)。db / store / embedder / dynamics を編成する。

全メソッドは JSON 化可能な値か models のデータ型を返す。時刻は time.time() を
使うが、テスト容易性のため now を引数で注入できるようにする(now: float | None)。
"""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from . import dynamics
from .config import Settings, get_settings
from .db import IndexDB
from .embedder import Embedder
from .models import MemoryRecord, RecallHit
from .store import MarkdownStore


class MemoryEngine:
    def __init__(self, settings: Settings, store: MarkdownStore, db: IndexDB,
                 embedder: Embedder) -> None:
        self.settings = settings
        self.store = store
        self.db = db
        self.embedder = embedder

    # ------------------------------------------------------------------ remember
    def remember(
        self,
        content: str,
        type: str,
        importance: int,
        *,
        tags: list[str] | None = None,
        source: str = "unknown",
        related_ids: list[str] | None = None,
        room: str | None = None,
        now: float | None = None,
    ) -> dict:
        """保存。手順:
        1. content を埋め込み、vector_search(top1, 同 type, tier=hot)で重複検知。
           cos >= settings.dup_threshold なら新規作成せず、既存記憶に
           reinforce 相当のイベントを記録して
           {"id": 既存id, "status": "duplicate_reinforced"} を返す。
        2. store.create → db.upsert_memory。
        3. create イベントを dynamics.create_event_weight(importance) の重みで記録
           (初期符号化ブースト)。
        4. related_ids へ explicit リンク(db + store 両方。store 側は双方向不要、
           新規側の frontmatter にのみ記録)。
        5. tags に "correction" が含まれる場合 importance を
           max(importance, settings.correction_min_importance) に引き上げる。
        返り値: {"id", "status": "created", "path"}。
        """
        ts = now if now is not None else time.time()
        tags = list(tags) if tags else []
        room = room or "common"

        # correction タグがある場合 importance を引き上げる
        if "correction" in tags:
            importance = max(importance, self.settings.correction_min_importance)

        # 1. 重複検知: 同タイプ・同部屋・tier=hot で top1 ベクトル検索
        #    (部屋を跨いだ併合は文脈分離を壊すので同部屋に限定する)
        vec = self.embedder.embed_docs([content])[0]
        candidates = self.db.vector_search(
            vec, 1, tiers=["hot"], types=[type], rooms=[room]
        )
        if candidates:
            top_id, top_cos = candidates[0]
            if top_cos >= self.settings.dup_threshold:
                # 重複とみなし、既存記憶に reinforce イベントを記録
                weight = self.settings.reinforce_weight
                self.db.add_event(top_id, "reinforce", weight, ts)
                return {"id": top_id, "status": "duplicate_reinforced"}

        # 2. store.create → db.upsert_memory
        record = self.store.create(
            content=content,
            type=type,
            importance=importance,
            tags=tags,
            source=source,
            links=related_ids or [],
            room=room,
        )

        self.db.upsert_memory(
            id=record.id,
            path=str(record.path),
            type=record.type,
            content_hash=record.content_hash,
            created_at=ts,
            importance=record.importance,
            tier=record.tier,
            content=content,
            embedding=vec,
            room=record.room,
        )

        # 3. create イベントを初期符号化ブーストで記録
        create_weight = dynamics.create_event_weight(
            importance, alpha=self.settings.create_alpha
        )
        self.db.add_event(record.id, "create", create_weight, ts)

        # 4. explicit リンク(db + store)
        if related_ids:
            for rel_id in related_ids:
                self.db.add_link(record.id, rel_id, "explicit", increment=1.0,
                                 max_weight=1.0)
                # store 側は新規側の frontmatter のみ(双方向不要)
                # store.create 時に links を渡してあるので追加更新は不要

        return {
            "id": record.id,
            "status": "created",
            "path": str(record.path),
        }

    # -------------------------------------------------------------------- recall
    def recall(
        self,
        query: str,
        *,
        mode: str = "fast",        # "fast" | "deep" | "exhaustive"
        limit: int = 5,
        type: str | None = None,
        room: str | None = None,   # None/"*"=全部屋。指定時は {room, common} に限定
        now: float | None = None,
        record_hits: bool = True,
    ) -> dict:
        """検索。fast の手順:
        1. tier=hot(episode は除外。type 指定があればそれを優先)で
           vector_search top-candidate_k と keyword_search top-candidate_k。
        2. dynamics.rrf_merge で統合 → 上位 ~candidate_k 件を候補に。
        3. 候補の relevance はベクトル類似度(FTS のみヒットは候補中の最小類似度を
           割り当てる)。db.get_events で活性度を計算し dynamics.final_score で再ランク。
           decay は dynamics.decay_rate(importance)。
        4. 上位 limit 件を RecallHit で返し、record_hits なら recall_hit イベント
           (weight=settings.recall_hit_weight)を記録。
        5. 最高スコア < settings.deep_score_threshold なら deep を自動発動して
           統合し、結果に "auto_deepened": True を付ける。

        deep の追加手順:
        - tier に cold / superseded、type に episode も含めて 1 をやり直す。
        - fast 上位をシードに db.get_links(kinds=co_recall/explicit/derived_from)
          から隣接関数を作り dynamics.spread で拡散。リンク経由のみで到達した
          記憶は via="associative"。relevance はクエリとの実コサインを別途計算。
        - superseded の記憶は note に「→ [後継id] により訂正済み」を入れ、
          後継(superseded_by リンク先)も結果に含める。

        exhaustive(網羅検索)の手順:
        - 候補数を絞らず全 tier/type の全記憶について、クエリとのコサイン類似
          (relevance)のみで順位付けする。活性度は同点時のタイブレークのみ。
        - 長く使われず沈んだ(活性度の低い)記憶でも意味的に近ければ必ず浮上する。
        - settings.exhaustive_min_relevance 未満は返さない。deep の最高スコアが
          settings.exhaustive_score_threshold 未満のときは自動で exhaustive へ
          エスカレーションする(その場合 mode に "exhaustive" が返る)。
        返り値: {"hits": [RecallHit を dict 化], "mode", "auto_deepened": bool}。
        """
        ts = now if now is not None else time.time()
        auto_deepened = False

        # 部屋フィルタ: 指定された部屋 + 共通(common)だけを見る
        rooms: list[str] | None = None
        if room is not None and room != "*":
            rooms = sorted({room, "common"})

        if mode == "exhaustive":
            # 明示的な網羅検索: 活性度を無視し関連度のみで全件から拾う
            hits = self._exhaustive_recall(query, limit=limit, type=type,
                                           rooms=rooms, now=ts)
        else:
            # fast モードでの検索
            hits, best_score = self._fast_recall(query, limit=limit, type=type,
                                                 rooms=rooms, now=ts)

            # 最高スコアが閾値未満なら deep を自動発動
            if mode == "fast" and best_score < self.settings.deep_score_threshold:
                mode = "deep"
                auto_deepened = True

            if mode == "deep":
                hits = self._deep_recall(query, fast_hits=hits, limit=limit,
                                         type=type, rooms=rooms, now=ts)
                # deep でも最高スコアが弱い=沈んだ記憶や candidate_k の枠外で
                # 掘りきれていない可能性。関連度のみの網羅検索を試し、より関連度の
                # 高い結果が得られたときだけ採用する(空振りで結果を劣化させない)。
                deep_best = hits[0].score if hits else 0.0
                if deep_best < self.settings.exhaustive_score_threshold:
                    ex_hits = self._exhaustive_recall(
                        query, limit=limit, type=type, rooms=rooms, now=ts)
                    deep_best_rel = max((h.relevance for h in hits), default=0.0)
                    ex_best_rel = max((h.relevance for h in ex_hits), default=0.0)
                    if ex_hits and ex_best_rel > deep_best_rel:
                        mode = "exhaustive"
                        auto_deepened = True
                        hits = ex_hits

        # recall_hit イベントを記録
        if record_hits:
            for hit in hits:
                self.db.add_event(hit.id, "recall_hit",
                                  self.settings.recall_hit_weight, ts)

        return {
            "hits": [_hit_to_dict(h) for h in hits],
            "mode": mode,
            "auto_deepened": auto_deepened,
        }

    def _fast_recall(
        self,
        query: str,
        *,
        limit: int,
        type: str | None,
        rooms: list[str] | None = None,
        now: float,
    ) -> tuple[list[RecallHit], float]:
        """fast recall の内部実装。(hits, best_score) を返す。"""
        s = self.settings
        k = s.candidate_k

        # episode は fast では除外
        if type is not None:
            search_types = [type]
        else:
            search_types = ["knowledge", "preference", "project"]

        search_tiers = ["hot"]

        # クエリベクトルを生成
        qvec = self.embedder.embed_query(query)

        # ベクトル検索 + キーワード検索
        vec_results = self.db.vector_search(
            qvec, k, tiers=search_tiers, types=search_types, rooms=rooms
        )
        kw_results = self.db.keyword_search(
            query, k, tiers=search_tiers, types=search_types, rooms=rooms
        )

        # ベクトル類似度マップ
        vec_sim: dict[str, float] = {id_: sim for id_, sim in vec_results}
        # FTS スコアマップ(BM25 は小さいほど良いので順位リスト用)
        vec_ids = [id_ for id_, _ in vec_results]
        kw_ids = [id_ for id_, _ in kw_results]

        # RRF で統合
        merged = dynamics.rrf_merge([vec_ids, kw_ids], k=s.rrf_k)

        # 候補の relevance: ベクトル類似度。FTS のみヒットには候補中の最小類似度を割当
        min_vec_sim = min(vec_sim.values()) if vec_sim else 0.0
        relevances: dict[str, float] = {}
        for id_ in merged:
            if id_ in vec_sim:
                relevances[id_] = vec_sim[id_]
            else:
                relevances[id_] = min_vec_sim

        # 活性度を計算して最終スコアで再ランク
        candidate_ids = list(merged.keys())
        events_map = self.db.get_events(candidate_ids)
        # importance を取得
        mem_rows = {m["id"]: m for m in self.db.all_memories(
            tiers=search_tiers, types=search_types, rooms=rooms)}

        scored: list[tuple[float, str]] = []
        for id_ in candidate_ids:
            mem = mem_rows.get(id_)
            if mem is None:
                continue
            imp = mem["importance"]
            d = dynamics.decay_rate(imp)
            act = dynamics.activation_norm(events_map.get(id_, []), now, d,
                                           min_elapsed=s.min_elapsed_seconds)
            rel = relevances[id_]
            score = dynamics.final_score(
                rel, act, imp,
                w_relevance=s.w_relevance,
                w_activation=s.w_activation,
                w_importance=s.w_importance,
            )
            scored.append((score, id_))

        scored.sort(reverse=True)
        top = scored[:limit]

        # RecallHit を組み立て(content は path から読む)
        hits: list[RecallHit] = []
        best_score = 0.0
        for score, id_ in top:
            mem = mem_rows[id_]
            imp = mem["importance"]
            d = dynamics.decay_rate(imp)
            act = dynamics.activation_norm(events_map.get(id_, []), now, d,
                                           min_elapsed=s.min_elapsed_seconds)
            rel = relevances[id_]
            # content を store から取得
            try:
                rec = self.store.read(Path(mem["path"]))
                content = rec.content
                tags = rec.tags
                tier = rec.tier
            except Exception:
                content = ""
                tags = []
                tier = mem.get("tier", "hot")
            hit = RecallHit(
                id=id_,
                content=content,
                type=mem["type"],
                tags=tags,
                tier=tier,
                score=score,
                relevance=rel,
                activation=act,
                importance=imp / 10.0,
                via="direct",
                room=mem.get("room", "common"),
            )
            hits.append(hit)
            if score > best_score:
                best_score = score

        return hits, best_score

    def _deep_recall(
        self,
        query: str,
        *,
        fast_hits: list[RecallHit],
        limit: int,
        type: str | None,
        rooms: list[str] | None = None,
        now: float,
    ) -> list[RecallHit]:
        """deep recall: tier=cold/superseded・episode も含め再検索 + 拡散活性化。"""
        s = self.settings
        k = s.candidate_k

        # deep は全 tier、全 type を対象(部屋フィルタは維持する)
        search_tiers = ["hot", "cold", "superseded"]
        search_types = [type] if type is not None else None

        qvec = self.embedder.embed_query(query)

        vec_results = self.db.vector_search(
            qvec, k, tiers=search_tiers, types=search_types, rooms=rooms
        )
        kw_results = self.db.keyword_search(
            query, k, tiers=search_tiers, types=search_types, rooms=rooms
        )

        vec_sim: dict[str, float] = {id_: sim for id_, sim in vec_results}
        min_vec_sim = min(vec_sim.values()) if vec_sim else 0.0

        vec_ids = [id_ for id_, _ in vec_results]
        kw_ids = [id_ for id_, _ in kw_results]
        merged = dynamics.rrf_merge([vec_ids, kw_ids], k=s.rrf_k)

        # シード: fast_hits の上位
        seed_map: dict[str, float] = {h.id: h.score for h in fast_hits}

        # リンクから隣接関数を構築(co_recall/explicit/derived_from)
        all_candidate_ids = list(set(list(merged.keys()) + list(seed_map.keys())))
        link_rows = self.db.get_links(
            all_candidate_ids,
            kinds=["co_recall", "explicit", "derived_from"]
        )
        # 双方向の隣接グラフ
        adjacency: dict[str, list[tuple[str, float]]] = {}
        for src, dst, kind, weight in link_rows:
            adjacency.setdefault(src, []).append((dst, weight))
            adjacency.setdefault(dst, []).append((src, weight))

        def neighbors(id_: str):
            return adjacency.get(id_, [])

        # 拡散活性化
        spread_scores = dynamics.spread(
            seed_map, neighbors,
            max_hops=s.max_hops,
            hop_decay=s.hop_decay,
        )

        # 全候補 = merged + 拡散で到達したノード
        all_ids = set(merged.keys()) | set(spread_scores.keys())
        # 直接候補 vs 連想経由を区別
        direct_ids = set(merged.keys())

        # 連想経由ノードの relevance: 実コサインを計算
        assoc_ids = all_ids - direct_ids
        if assoc_ids:
            emb_map = self.db.get_embeddings(list(assoc_ids))
            for id_, emb in emb_map.items():
                cos = float(np.dot(qvec, emb))
                vec_sim[id_] = max(0.0, cos)

        # 全候補の importance を取得。rooms フィルタ付きなので、拡散活性化で
        # 他の部屋に到達してもここで弾かれる(連想経由の部屋漏れ防止)
        all_mem_rows = {m["id"]: m for m in self.db.all_memories(
            tiers=search_tiers, types=search_types, rooms=rooms
        )}

        events_map = self.db.get_events(list(all_ids))

        # superseded 記憶の後継マップ
        superseded_links = self.db.get_links(
            list(all_ids), kinds=["superseded_by"]
        )
        successor_map: dict[str, str] = {}
        for src, dst, kind, _ in superseded_links:
            if kind == "superseded_by":
                successor_map[src] = dst

        scored: list[tuple[float, str, str, float, float]] = []  # (score, id, via, rel, act)
        for id_ in all_ids:
            mem = all_mem_rows.get(id_)
            if mem is None:
                continue
            via = "direct" if id_ in direct_ids else "associative"
            imp = mem["importance"]
            d = dynamics.decay_rate(imp)
            act = dynamics.activation_norm(events_map.get(id_, []), now, d,
                                           min_elapsed=s.min_elapsed_seconds)
            rel = vec_sim.get(id_, min_vec_sim)
            if via == "associative":
                # 連想経由ノードはクエリとの直接類似が低いからこそリンクで
                # 辿られている。強いリンクで繋がっていること自体が関連性の
                # 証拠なので、拡散活性化の伝播スコアで関連度を底上げする
                # (これが無いと連想記憶がノイズに埋もれて永久に出てこない)
                rel = max(rel, spread_scores.get(id_, 0.0))
            score = dynamics.final_score(
                rel, act, imp,
                w_relevance=s.w_relevance,
                w_activation=s.w_activation,
                w_importance=s.w_importance,
            )
            scored.append((score, id_, via, rel, act))

        scored.sort(reverse=True)
        top = scored[:limit]

        hits: list[RecallHit] = []
        for score, id_, via, rel, act in top:
            mem = all_mem_rows[id_]
            imp = mem["importance"]
            try:
                rec = self.store.read(Path(mem["path"]))
                content = rec.content
                tags = rec.tags
                tier = rec.tier
            except Exception:
                content = ""
                tags = []
                tier = mem.get("tier", "hot")

            # superseded 記憶には note を付与
            note = ""
            if tier == "superseded" and id_ in successor_map:
                note = f"→ [{successor_map[id_]}] により訂正済み"

            hit = RecallHit(
                id=id_,
                content=content,
                type=mem["type"],
                tags=tags,
                tier=tier,
                score=score,
                relevance=rel,
                activation=act,
                importance=imp / 10.0,
                via=via,
                note=note,
                room=mem.get("room", "common"),
            )
            hits.append(hit)

        return hits

    def _exhaustive_recall(
        self,
        query: str,
        *,
        limit: int,
        type: str | None,
        rooms: list[str] | None = None,
        now: float,
    ) -> list[RecallHit]:
        """網羅検索: 活性度を無視し関連度のみで全 tier/type を総当たりする。

        fast/deep は final_score に活性度が効くため、長く使われず沈んだ記憶は
        関連が高くても limit の外へ押し出される(「忘れない記憶」なのに想起
        できない)。ここでは候補数を絞らず全記憶のクエリとのコサイン類似だけで
        順位付けするので、沈んだ記憶も意味的に近ければ必ず浮上する。
        部屋フィルタは維持し、settings.exhaustive_min_relevance 未満は返さない。
        """
        s = self.settings

        # 全 tier・全 type を対象(部屋フィルタは維持する)
        search_tiers = ["hot", "cold", "superseded"]
        search_types = [type] if type is not None else None

        mem_rows = self.db.all_memories(
            tiers=search_tiers, types=search_types, rooms=rooms
        )
        if not mem_rows:
            return []

        mem_by_id = {m["id"]: m for m in mem_rows}
        ids = list(mem_by_id.keys())

        qvec = self.embedder.embed_query(query)
        emb_map = self.db.get_embeddings(ids)
        events_map = self.db.get_events(ids)

        # superseded 記憶の後継マップ(訂正済み注記用)
        successor_map: dict[str, str] = {}
        for src, dst, kind, _ in self.db.get_links(ids, kinds=["superseded_by"]):
            if kind == "superseded_by":
                successor_map[src] = dst

        scored: list[tuple[float, float, str]] = []  # (relevance, activation, id)
        for id_ in ids:
            emb = emb_map.get(id_)
            if emb is None:
                continue
            rel = max(0.0, float(np.dot(qvec, emb)))
            if rel < s.exhaustive_min_relevance:
                continue
            imp = mem_by_id[id_]["importance"]
            d = dynamics.decay_rate(imp)
            act = dynamics.activation_norm(events_map.get(id_, []), now, d,
                                           min_elapsed=s.min_elapsed_seconds)
            scored.append((rel, act, id_))

        # 関連度を主、活性度は同点時のタイブレークのみ(沈んでいても拾う)
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        top = scored[:limit]

        hits: list[RecallHit] = []
        for rel, act, id_ in top:
            mem = mem_by_id[id_]
            imp = mem["importance"]
            try:
                rec = self.store.read(Path(mem["path"]))
                content = rec.content
                tags = rec.tags
                tier = rec.tier
            except Exception:
                content = ""
                tags = []
                tier = mem.get("tier", "hot")

            note = ""
            if tier == "superseded" and id_ in successor_map:
                note = f"→ [{successor_map[id_]}] により訂正済み"

            hits.append(RecallHit(
                id=id_,
                content=content,
                type=mem["type"],
                tags=tags,
                tier=tier,
                score=rel,            # 網羅検索の順位基準は関連度そのもの
                relevance=rel,
                activation=act,
                importance=imp / 10.0,
                via="exhaustive",
                note=note,
                room=mem.get("room", "common"),
            ))

        return hits

    # ----------------------------------------------------------------- reinforce
    def reinforce(self, ids: list[str], *, strength: float = 1.0,
                  now: float | None = None) -> dict:
        """使用報告。各 id に reinforce イベント
        (weight = settings.reinforce_weight * clamp(strength, 0.1, reinforce_strength_max))。
        同時に reinforce された id ペア全てに co_recall リンクを
        increment=settings.colink_increment で強化(ヘッブ則)。
        存在しない id は無視して結果に "unknown_ids" として列挙。
        """
        ts = now if now is not None else time.time()
        s = self.settings

        # strength をクランプ
        clamped = max(0.1, min(strength, s.reinforce_strength_max))
        weight = s.reinforce_weight * clamped

        reinforced: list[str] = []
        unknown: list[str] = []
        for id_ in ids:
            mem = self.db.get_memory(id_)
            if mem is None:
                unknown.append(id_)
                continue
            self.db.add_event(id_, "reinforce", weight, ts)
            reinforced.append(id_)

        # ヘッブ則: 同時に reinforce された ペアに co_recall リンク
        for i in range(len(reinforced)):
            for j in range(i + 1, len(reinforced)):
                self.db.add_link(
                    reinforced[i], reinforced[j], "co_recall",
                    increment=s.colink_increment,
                    max_weight=s.colink_max,
                )
                self.db.add_link(
                    reinforced[j], reinforced[i], "co_recall",
                    increment=s.colink_increment,
                    max_weight=s.colink_max,
                )

        return {
            "reinforced": reinforced,
            "unknown_ids": unknown,
        }

    # ------------------------------------------------------------------- correct
    def correct(self, id: str, corrected_content: str, reason: str, *,
                source: str = "unknown", now: float | None = None) -> dict:
        """誤り訂正(ハイパーコレクション効果)。
        1. 旧記憶を取得(無ければ {"status": "not_found"})。
        2. 新記憶本文を組み立てる:
               {corrected_content}

               > [!note] 訂正の記録
               > 以前は「{旧本文の先頭200文字}」と誤認していた。
               > 訂正理由: {reason}
               > 旧記憶: [[旧id]]
        3. remember 相当で新規作成。type は旧記憶を継承、
           importance = max(旧importance, settings.correction_min_importance)、
           tags は旧 tags + "correction"。重複検知はスキップする。
        4. 旧記憶: tier=superseded(store+db)、superseded_by リンク(旧→新)。
        返り値: {"new_id", "old_id", "status": "corrected"}。
        """
        ts = now if now is not None else time.time()
        s = self.settings

        # 1. 旧記憶を取得
        old_mem = self.db.get_memory(id)
        if old_mem is None:
            return {"status": "not_found"}

        old_path = Path(old_mem["path"])
        try:
            old_record = self.store.read(old_path)
        except Exception:
            return {"status": "not_found"}

        old_content = old_record.content
        old_snippet = old_content[:200]

        # 2. 新記憶本文を組み立て
        new_content = (
            f"{corrected_content}\n\n"
            f"> [!note] 訂正の記録\n"
            f"> 以前は「{old_snippet}」と誤認していた。\n"
            f"> 訂正理由: {reason}\n"
            f"> 旧記憶: [[{id}]]"
        )

        # 3. 新規記憶作成(重複検知はスキップ: _remember_direct を使う)
        new_importance = max(old_record.importance, s.correction_min_importance)
        new_tags = list(old_record.tags) + ["correction"]
        if "correction" not in new_tags:
            new_tags.append("correction")

        new_result = self._remember_direct(
            content=new_content,
            type=old_record.type,
            importance=new_importance,
            tags=new_tags,
            source=source,
            related_ids=[id],
            room=old_record.room,
            now=ts,
        )
        new_id = new_result["id"]

        # 4. 旧記憶を superseded に降格
        self.store.set_tier(old_record, "superseded")
        self.db.set_tier(id, "superseded")

        # superseded_by リンク(旧→新)
        self.db.add_link(id, new_id, "superseded_by", increment=1.0, max_weight=1.0)

        return {
            "new_id": new_id,
            "old_id": id,
            "status": "corrected",
        }

    def _remember_direct(
        self,
        content: str,
        type: str,
        importance: int,
        *,
        tags: list[str] | None = None,
        source: str = "unknown",
        related_ids: list[str] | None = None,
        room: str = "common",
        now: float,
    ) -> dict:
        """重複検知をスキップした記憶の直接保存(correct から呼ぶ)。"""
        s = self.settings
        tags = list(tags) if tags else []

        vec = self.embedder.embed_docs([content])[0]

        record = self.store.create(
            content=content,
            type=type,
            importance=importance,
            tags=tags,
            source=source,
            links=related_ids or [],
            room=room,
        )

        self.db.upsert_memory(
            id=record.id,
            path=str(record.path),
            type=record.type,
            content_hash=record.content_hash,
            created_at=now,
            importance=record.importance,
            tier=record.tier,
            content=content,
            embedding=vec,
            room=record.room,
        )

        create_weight = dynamics.create_event_weight(
            importance, alpha=s.create_alpha
        )
        self.db.add_event(record.id, "create", create_weight, now)

        if related_ids:
            for rel_id in related_ids:
                self.db.add_link(record.id, rel_id, "explicit", increment=1.0,
                                 max_weight=1.0)

        return {
            "id": record.id,
            "status": "created",
            "path": str(record.path),
        }

    # --------------------------------------------------------------------- misc
    def link(self, src: str, dst: str) -> dict:
        """explicit リンクを張る(db 両方向ではなく src→dst の1エッジ + store の
        src frontmatter に追記)。"""
        # DB にリンクを追加
        self.db.add_link(src, dst, "explicit", increment=1.0, max_weight=1.0)

        # store の src frontmatter に追記
        src_mem = self.db.get_memory(src)
        if src_mem is not None:
            try:
                src_record = self.store.read(Path(src_mem["path"]))
                self.store.add_link(src_record, dst)
            except Exception:
                pass

        return {"src": src, "dst": dst, "status": "linked"}

    def forget(self, id: str) -> dict:
        """ソフト削除: store.move_to_trash + db では tier=trash(検索対象外)。"""
        mem = self.db.get_memory(id)
        if mem is None:
            return {"status": "not_found"}

        try:
            record = self.store.read(Path(mem["path"]))
            self.store.move_to_trash(record)
        except Exception:
            pass

        self.db.set_tier(id, "trash")
        return {"id": id, "status": "forgotten"}

    def stats(self) -> dict:
        return self.db.stats()

    # ------------------------------------------------------------- consolidation
    def consolidation_candidates(self, *, now: float | None = None) -> dict:
        """tier=hot で settings.consolidate_min_age_days より古い episode を取得し、
        埋め込みコサイン >= settings.consolidate_cluster_sim の貪欲法でクラスタ化。
        2件以上のクラスタのみ {"clusters": [{"ids": [...], "contents": [...]}]} で返す。
        要約自体は呼び出し元エージェント(LLM)が行う — サーバーは LLM を持たない。
        """
        ts = now if now is not None else time.time()
        s = self.settings
        min_age_seconds = s.consolidate_min_age_days * 86400.0

        # tier=hot、type=episode の記憶を取得
        all_mems = self.db.all_memories(tiers=["hot"], types=["episode"])

        # 古いものだけ絞る
        old_ids = []
        for mem in all_mems:
            age = ts - mem.get("created_at", ts)
            if age >= min_age_seconds:
                old_ids.append(mem["id"])

        if not old_ids:
            return {"clusters": []}

        # 埋め込みを取得してクラスタリング
        emb_map = self.db.get_embeddings(old_ids)
        # emb_map に含まれる id のみ有効
        valid_ids = [id_ for id_ in old_ids if id_ in emb_map]

        if len(valid_ids) < 2:
            return {"clusters": []}

        # 貪欲クラスタリング
        used = set()
        clusters: list[list[str]] = []
        for i, id_a in enumerate(valid_ids):
            if id_a in used:
                continue
            cluster = [id_a]
            used.add(id_a)
            vec_a = emb_map[id_a]
            for id_b in valid_ids[i + 1:]:
                if id_b in used:
                    continue
                vec_b = emb_map[id_b]
                cos = float(np.dot(vec_a, vec_b))
                if cos >= s.consolidate_cluster_sim:
                    cluster.append(id_b)
                    used.add(id_b)
            if len(cluster) >= 2:
                clusters.append(cluster)

        # 各クラスタの contents を取得
        mem_rows = {m["id"]: m for m in all_mems}
        result_clusters = []
        for cluster_ids in clusters:
            contents = []
            for id_ in cluster_ids:
                mem = mem_rows.get(id_)
                if mem is None:
                    continue
                try:
                    rec = self.store.read(Path(mem["path"]))
                    contents.append(rec.content)
                except Exception:
                    contents.append("")
            result_clusters.append({"ids": cluster_ids, "contents": contents})

        return {"clusters": result_clusters}

    def mark_consolidated(self, episode_ids: list[str], new_memory_id: str) -> dict:
        """統合完了処理: 各 episode に derived_from リンク(episode→new)を張り、
        tier=cold に降格。"""
        updated = []
        for ep_id in episode_ids:
            mem = self.db.get_memory(ep_id)
            if mem is None:
                continue
            # derived_from リンク(episode→new)
            self.db.add_link(ep_id, new_memory_id, "derived_from",
                             increment=1.0, max_weight=1.0)
            # tier=cold に降格
            self.db.set_tier(ep_id, "cold")
            try:
                rec = self.store.read(Path(mem["path"]))
                self.store.set_tier(rec, "cold")
            except Exception:
                pass
            updated.append(ep_id)

        return {
            "consolidated": updated,
            "new_memory_id": new_memory_id,
            "status": "ok",
        }

    # ------------------------------------------------------------------- reindex
    def reindex(self) -> dict:
        """Markdown 正本から DB を突き合わせて再構築:
        - store.scan_all() の各記録について content_hash を DB と比較、
          差異(手編集)や未登録は再埋め込みして upsert。
        - DB にあるがファイルが消えた id は delete_memory。
        - 件数を {"added", "updated", "removed", "unchanged"} で返す。
        """
        added = 0
        updated = 0
        unchanged = 0

        # store からファイル全走査
        seen_ids: set[str] = set()
        for record in self.store.scan_all():
            seen_ids.add(record.id)
            db_mem = self.db.get_memory(record.id)
            if db_mem is None:
                # 未登録: 埋め込んで upsert。created_at は正本 frontmatter から復元
                vec = self.embedder.embed_docs([record.content])[0]
                created_at = _created_to_epoch(record.created)
                self.db.upsert_memory(
                    id=record.id,
                    path=str(record.path),
                    type=record.type,
                    content_hash=record.content_hash,
                    created_at=created_at,
                    importance=record.importance,
                    tier=record.tier,
                    content=record.content,
                    embedding=vec,
                    room=record.room,
                )
                # イベントが無い(=ゼロから再構築した)場合は create イベントを
                # 再シードする。これが無いと再構築後の活性度が全件 0 になる
                if not self.db.get_events([record.id]).get(record.id):
                    self.db.add_event(
                        record.id,
                        "create",
                        dynamics.create_event_weight(
                            record.importance, alpha=self.settings.create_alpha
                        ),
                        created_at,
                    )
                added += 1
            elif (db_mem.get("content_hash", "") != record.content_hash
                  or db_mem.get("room", "common") != record.room):
                # 手編集で差異あり(本文または room): 再埋め込みして upsert
                vec = self.embedder.embed_docs([record.content])[0]
                self.db.upsert_memory(
                    id=record.id,
                    path=str(record.path),
                    type=record.type,
                    content_hash=record.content_hash,
                    created_at=db_mem.get("created_at", 0.0),
                    importance=record.importance,
                    tier=record.tier,
                    content=record.content,
                    embedding=vec,
                    room=record.room,
                )
                updated += 1
            else:
                unchanged += 1

        # DB にあるがファイルが消えた id を削除
        all_db_ids = {m["id"] for m in self.db.all_memories()}
        orphans = all_db_ids - seen_ids
        for orphan_id in orphans:
            self.db.delete_memory(orphan_id)
        removed = len(orphans)

        return {
            "added": added,
            "updated": updated,
            "removed": removed,
            "unchanged": unchanged,
        }

    # ------------------------------------------------ startup index freshness
    def check_index_freshness(self, *, mode: str = "auto") -> dict:
        """起動時のインデックス同期チェック(マルチマシン共有対策)。

        記憶 Markdown は共有(例: Google Drive)でも index.db はマシンごとローカルな
        ため、他マシンが書いた記憶が index に無く recall に一切出ない盲点が生じる。
        Markdown(非trash)のファイル数と index の active(hot/cold/superseded)件数を
        比較し、乖離があれば:
          - mode="auto": reindex して同期する
          - mode="warn": 警告情報を返す(呼び出し側がログ出力。書き込みはしない)
          - mode="off" : 何もしない
        返り値: {"action", "markdown", "index", ...}。action は
        "off"/"in_sync"/"warn"/"reindexed"。
        """
        if mode == "off":
            return {"action": "off"}

        md_count = self.store.count_memory_files()
        idx_count = len(self.db.all_memories(
            tiers=["hot", "cold", "superseded"]
        ))

        if md_count == idx_count:
            # 速い経路: raw .md 件数が一致すれば同期済みとみなし、走査しない
            return {"action": "in_sync", "markdown": md_count, "index": idx_count}

        # 件数が違う。空ファイル・壊れた frontmatter・id 無しの非記憶 .md による
        # 見かけ上のズレかもしれない(scan_all はそれらをスキップする)。無駄な
        # reindex を避けるため、index と同じ「有効な記憶」母集団で正確に数え直す。
        valid_count = sum(1 for _ in self.store.scan_all())
        if valid_count == idx_count:
            return {
                "action": "in_sync",
                "markdown": md_count,
                "index": idx_count,
                "valid": valid_count,
                "note": "raw .md 件数差は空/壊れた/非記憶 md による見かけ上のもの",
            }

        if mode == "warn":
            return {
                "action": "warn",
                "markdown": md_count,
                "index": idx_count,
                "valid": valid_count,
                "drift": valid_count - idx_count,
            }

        # auto: reindex して同期(他マシンの未取り込み記憶を index へ取り込む)
        reindex_result = self.reindex()
        return {
            "action": "reindexed",
            "markdown": md_count,
            "index": idx_count,
            "valid": valid_count,
            "reindex": reindex_result,
        }


def build_engine(settings: Settings | None = None, *, embedder: Embedder | None = None
                 ) -> MemoryEngine:
    """既定構成でエンジンを組み立てるファクトリ(server / cli から使う)。
    embedder 未指定なら RuriEmbedder(settings.embed_model)。"""
    if settings is None:
        settings = get_settings()

    if embedder is None:
        from .embedder import RuriEmbedder
        embedder = RuriEmbedder(
            model_name=settings.embed_model,
            query_prefix=settings.query_prefix,
            doc_prefix=settings.doc_prefix,
        )

    store = MarkdownStore(settings.memories_dir)
    # 次元は必ず実モデルから取得する(推定が外れると DB のベクトル表が
    # 誤った次元で固定され、以後の全埋め込みが壊れる)。RuriEmbedder の
    # 初回ロードはここで走るが、stdio サーバーは常駐なので一度きり。
    db = IndexDB(settings.db_path, embedder.dim)

    return MemoryEngine(settings=settings, store=store, db=db, embedder=embedder)


def _created_to_epoch(created: str) -> float:
    """frontmatter の created(ISO 8601)を unix 秒へ。壊れていれば現在時刻。"""
    from datetime import datetime

    try:
        return datetime.fromisoformat(created).timestamp()
    except (ValueError, TypeError):
        return time.time()


def _hit_to_dict(hit: RecallHit) -> dict:
    """RecallHit を JSON 化可能な dict に変換。"""
    return {
        "id": hit.id,
        "content": hit.content,
        "type": hit.type,
        "tags": hit.tags,
        "tier": hit.tier,
        "score": hit.score,
        "relevance": hit.relevance,
        "activation": hit.activation,
        "importance": hit.importance,
        "via": hit.via,
        "note": hit.note,
        "room": hit.room,
    }
