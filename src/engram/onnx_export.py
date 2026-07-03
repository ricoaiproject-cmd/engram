"""ONNX エクスポート(`engram export-onnx`)。

sentence-transformers(torch)で動いている埋め込みモデルを ONNX へ一度だけ
変換し、以後のランタイムを onnxruntime + tokenizers だけにする。torch の
import(warm 12〜24秒 / cold 50秒超)が起動経路から消えるのが目的。

変換は torch.onnx.export を直接使う。optimum を使わないのは意図的:
optimum は transformers のバージョン上限が厳しく、導入すると既存環境の
transformers を巻き戻して sentence-transformers 本体を壊すことがある
(2026-07-03 に transformers 5.12→4.57 への降格で実際に発生)。torch は
sentence-transformers 経由で既に必須依存なので、追加依存ゼロで変換できる。

安全装置: 変換後に torch 経路(RuriEmbedder)と ONNX 経路(OnnxRuriEmbedder)で
同じテキスト群を埋め込み、コサイン類似の最小値が PARITY_MIN 未満なら失敗として
モデルディレクトリごと破棄する。DB は次元とベクトル分布をモデルに固定している
(db.py の dim mismatch 例外)ため、分布がずれた ONNX を黙って採用すると既存の
全記憶の検索が静かに壊れる。
"""

from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings

# torch fp32 → ONNX fp32 の変換誤差は通常 1e-6 級。0.999 を下回るのは
# 変換の仕方が違う(pooling / prefix / tokenizer 差異)ことを意味する。
PARITY_MIN = 0.999

# パリティ検証用のサンプル。実運用の記憶に近い、長さと内容が雑多な日本語文。
# 長さを散らしてあるのは、トレース時に系列長依存の分岐が焼き付いた場合に
# 検証で確実に露見させるため。
PARITY_TEXTS = [
    "Windows では MCP サーバーのスレッドで torch を import すると劣化する",
    "ユーザーの好み: 業務文書は和暦・右詰めヘッダーで作成すること",
    "pip の再インストール失敗で site-packages に残骸が残り import 不能になった",
    "令和8年度 研究データ基盤開発委員会 今後の計画書",
    "engram recall のクエリ",
    "短い文",
    "SQLite の仮想テーブル vec0 は埋め込み次元を作成時に固定する。"
    "次元が変わると既存 DB は開けず、reindex 以前に DB の作り直しが必要になる。"
    "運用上は export-onnx のパリティ検証がこの事故を防ぐ最後の砦になる。",
    "会議は毎週火曜 10 時から。議事録は MeetingRecords フォルダに保存する。",
    # ModernBERT は 128 トークン超でスライディングウィンドウ注意に切り替わる。
    # その経路がトレースに正しく乗ったかは長文でしか検証できないため、
    # 境界を大きく越える文を必ず1つ含める(~400トークン)。
    "長期記憶の設計では、意味(埋め込み)と思い出しやすさ(活性度)を分離する。"
    "埋め込みベクトルは固定し、検索順位は使用履歴に基づく活性度で変調する。"
    "使うほど活性化し、放置するとべき乗則で減衰するが、完全には消えない。"
    "印象的な文脈で符号化された記憶は減衰が遅く、訂正された誤りは間違えた経験ごと"
    "高い活性で刻み直される。これらの力学は ACT-R の宣言的記憶モジュールに由来し、"
    "実装では各記憶のアクセスイベント列から活性度を都度計算する。検索時は関連度・"
    "活性度・重要度の加重和で最終スコアを決め、重複検知にはコサイン類似の閾値を使う。"
    "この閾値は同一話題の短い日本語文で誤併合が起きた実例に基づいて調整された。"
    "多マシン運用では記憶の正本を Markdown として共有し、インデックスはマシンごとに"
    "ローカルへ置く。起動時に件数の乖離を検知して自動的に再インデックスする。",
]


def _resolve_pad_token(model_name: str, target_dir: Path) -> tuple[str, int]:
    """tokenizer.json を target_dir に配置し、(pad_token, pad_token_id) を返す。

    HF Hub のキャッシュから直接取得する。AutoTokenizer を経由しないのは、
    モデルによっては sentencepiece の slow 経路に迷い込んで失敗するため
    (ランタイムが使うのも tokenizers ライブラリ + tokenizer.json のみ)。
    pad token は special_tokens_map.json の定義を読む。
    """
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok_path = Path(hf_hub_download(model_name, "tokenizer.json"))
    shutil.copy(tok_path, target_dir / "tokenizer.json")

    pad_token = None
    try:
        stm_path = Path(hf_hub_download(model_name, "special_tokens_map.json"))
        with stm_path.open(encoding="utf-8") as f:
            entry = json.load(f).get("pad_token")
        if isinstance(entry, dict):
            pad_token = entry.get("content")
        elif isinstance(entry, str):
            pad_token = entry
    except Exception:
        pass
    if not pad_token:
        raise RuntimeError("pad token が特定できないモデルは未対応です")

    pad_id = Tokenizer.from_file(str(target_dir / "tokenizer.json")).token_to_id(
        pad_token
    )
    if pad_id is None:
        raise RuntimeError(f"pad token {pad_token!r} が語彙に存在しません")
    return pad_token, int(pad_id)


def _export_transformer(st_model, out_path: Path) -> None:
    """SentenceTransformer が抱える transformer 本体を ONNX ファイルへ書き出す。

    入力は (input_ids, attention_mask)、出力は last_hidden_state のみ。
    pooling と正規化は ONNX に含めない(embedder.mean_pool_normalize が担う)。
    batch / seq の両軸を動的にする。
    """
    import torch

    auto_model = st_model[0].auto_model
    auto_model.eval()

    class _LastHidden(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids, attention_mask):
            return self.m(
                input_ids=input_ids, attention_mask=attention_mask
            ).last_hidden_state

    wrapper = _LastHidden(auto_model)
    # パディングを含むサンプル入力(mask の 0 経路もトレースに乗せる)
    ids = torch.ones((2, 16), dtype=torch.int64)
    mask = torch.ones((2, 16), dtype=torch.int64)
    mask[1, 8:] = 0

    try:
        # 新エクスポータ(dynamo)。現代的なモデル(ModernBERT 等)はこちらが確実
        batch = torch.export.Dim("batch")
        seq = torch.export.Dim("seq")
        program = torch.onnx.export(
            wrapper,
            (ids, mask),
            dynamo=True,
            dynamic_shapes={
                "input_ids": {0: batch, 1: seq},
                "attention_mask": {0: batch, 1: seq},
            },
        )
        program.save(str(out_path))
    except Exception as e:
        print(
            f"[export-onnx] dynamo エクスポータ失敗、旧エクスポータで再試行: {e}",
            file=sys.stderr,
        )
        torch.onnx.export(
            wrapper,
            (ids, mask),
            str(out_path),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "last_hidden_state": {0: "batch", 1: "seq"},
            },
            dynamo=False,
        )


def export_onnx(settings: Settings, *, force: bool = False) -> dict:
    """settings.embed_model を ONNX 化して settings.onnx_model_dir に置く。

    戻り値はレポート dict(dim, パリティ統計, 出力先など)。失敗時は例外。
    """
    target = settings.onnx_model_dir
    if target.is_dir() and not force:
        raise FileExistsError(
            f"既に存在します: {target}\n上書きするには --force を付けてください"
        )

    tmp = target.with_name(target.name + ".tmp")
    if tmp.is_dir():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    try:
        from .embedder import OnnxRuriEmbedder, RuriEmbedder

        print(
            f"[1/4] {settings.embed_model} を torch でロード中(基準経路)...",
            file=sys.stderr,
        )
        ref = RuriEmbedder(
            model_name=settings.embed_model,
            query_prefix=settings.query_prefix,
            doc_prefix=settings.doc_prefix,
        )
        st_model = ref._load()
        ref_docs = ref.embed_docs(PARITY_TEXTS)
        ref_query = ref.embed_query(PARITY_TEXTS[0])
        dim = ref.dim
        max_seq = int(getattr(st_model, "max_seq_length", 8192))

        print("[2/4] ONNX へ変換中...", file=sys.stderr)
        _export_transformer(st_model, tmp / "model.onnx")
        pad_token, pad_token_id = _resolve_pad_token(settings.embed_model, tmp)

        meta = {
            "source_model": settings.embed_model,
            "dim": dim,
            "max_seq_length": max_seq,
            "pad_token_id": pad_token_id,
            "pad_token": pad_token,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        with (tmp / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print("[3/4] ONNX 経路でパリティ検証中...", file=sys.stderr)
        onnx = OnnxRuriEmbedder(
            tmp,
            query_prefix=settings.query_prefix,
            doc_prefix=settings.doc_prefix,
        )
        if onnx.dim != dim:
            raise RuntimeError(f"次元不一致: torch={dim}, onnx={onnx.dim}")
        onnx_docs = onnx.embed_docs(PARITY_TEXTS)
        onnx_query = onnx.embed_query(PARITY_TEXTS[0])

        # 両経路とも L2 正規化済みなので内積 = コサイン類似
        doc_cos = (ref_docs * onnx_docs).sum(axis=1)
        query_cos = float((ref_query * onnx_query).sum())
        min_cos = float(min(doc_cos.min(), query_cos))
        if min_cos < PARITY_MIN:
            raise RuntimeError(
                f"パリティ検証に失敗: min cosine = {min_cos:.6f} < {PARITY_MIN}\n"
                "ONNX 経路の分布が torch 経路とずれています。このモデルを採用すると"
                "既存 index.db の検索が壊れるため中止しました。"
            )

        meta["parity"] = {
            "n_texts": len(PARITY_TEXTS),
            "min_cosine": min_cos,
            "threshold": PARITY_MIN,
        }
        with (tmp / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        print("[4/4] 配置中...", file=sys.stderr)
        if target.is_dir():
            shutil.rmtree(target)
        tmp.replace(target)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    return {
        "model": settings.embed_model,
        "dim": dim,
        "max_seq_length": max_seq,
        "min_cosine": min_cos,
        "target": str(target),
        "onnx_size_mb": round(
            (target / "model.onnx").stat().st_size / 1024 / 1024, 1
        ),
    }
