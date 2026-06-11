# engram インストールスクリプト
# 使い方: irm <URL> | iex
#         または: .\install.ps1
#         または: .\install.ps1 -Source "C:\path\to\engram"
#
# Windows PowerShell 5.1 互換

param(
    [string]$Source = "git+https://github.com/ricoaiproject-cmd/engram.git"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================"
Write-Host " engram インストーラ"
Write-Host "========================================"
Write-Host ""

# ----------------------------------------------------------------
# Step 1: uv の確認とインストール
# ----------------------------------------------------------------
Write-Host "[1/4] uv のインストール確認..."

$uvPath = $null
try {
    $uvPath = (Get-Command uv -ErrorAction SilentlyContinue).Source
} catch {}

if ($uvPath) {
    Write-Host "  uv は既にインストール済みです: $uvPath"
} else {
    Write-Host "  uv が見つかりません。インストールします..."
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    } catch {
        Write-Host ""
        Write-Host "[エラー] uv のインストールに失敗しました。"
        Write-Host "  手動でインストールしてください: https://docs.astral.sh/uv/getting-started/installation/"
        exit 1
    }

    # 現セッションの PATH に uv を追加
    $uvBin = Join-Path $env:USERPROFILE ".local\bin"
    if ($env:PATH -notlike "*$uvBin*") {
        $env:PATH = "$uvBin;$env:PATH"
    }

    # 再確認
    try {
        $uvPath = (Get-Command uv -ErrorAction SilentlyContinue).Source
    } catch {}

    if (-not $uvPath) {
        Write-Host ""
        Write-Host "[エラー] uv のインストール後もコマンドが見つかりません。"
        Write-Host "  ターミナルを再起動してから再度お試しください。"
        exit 1
    }
    Write-Host "  uv のインストール完了: $uvPath"
}

Write-Host ""

# ----------------------------------------------------------------
# Step 2: git の確認とインストール(git+ ソースの取得に必要)
# ----------------------------------------------------------------
Write-Host "[2/4] git のインストール確認..."

if ($Source -like "git+*") {
    $gitPath = $null
    try {
        $gitPath = (Get-Command git -ErrorAction SilentlyContinue).Source
    } catch {}

    if ($gitPath) {
        Write-Host "  git は既にインストール済みです: $gitPath"
    } else {
        Write-Host "  git が見つかりません。インストールします..."
        winget install --id Git.Git -e --silent --accept-source-agreements --accept-package-agreements
        if (-not $?) {
            Write-Host ""
            Write-Host "[エラー] git のインストールに失敗しました。"
            Write-Host "  手動でインストールしてください: https://git-scm.com/downloads/win"
            exit 1
        }

        # 現セッションの PATH に git を追加
        $gitBin = "C:\Program Files\Git\cmd"
        if ((Test-Path $gitBin) -and ($env:PATH -notlike "*$gitBin*")) {
            $env:PATH = "$gitBin;$env:PATH"
        }

        try {
            $gitPath = (Get-Command git -ErrorAction SilentlyContinue).Source
        } catch {}

        if (-not $gitPath) {
            Write-Host ""
            Write-Host "[エラー] git のインストール後もコマンドが見つかりません。"
            Write-Host "  ターミナルを再起動してから再度お試しください。"
            exit 1
        }
        Write-Host "  git のインストール完了: $gitPath"
    }
} else {
    Write-Host "  ローカルソースのため git は不要です(スキップ)"
}

Write-Host ""

# ----------------------------------------------------------------
# Step 3: engram のインストール
# ----------------------------------------------------------------
Write-Host "[3/4] engram をインストールします..."
Write-Host "  ソース: $Source"
Write-Host "  (初回は Python 3.12 のダウンロードが発生する場合があります)"
Write-Host ""

uv tool install --python 3.12 --force $Source

if (-not $?) {
    Write-Host ""
    Write-Host "[エラー] engram のインストールに失敗しました。"
    Write-Host "  - Source の指定を確認してください: $Source"
    Write-Host "  - ネットワーク接続を確認してください"
    exit 1
}

# uv tool の shim PATH を追加
$uvToolBin = Join-Path $env:USERPROFILE ".local\bin"
if ($env:PATH -notlike "*$uvToolBin*") {
    $env:PATH = "$uvToolBin;$env:PATH"
}

Write-Host ""
Write-Host "  engram のインストール完了"
Write-Host ""

# ----------------------------------------------------------------
# Step 4: セットアップウィザードの実行
# ----------------------------------------------------------------
Write-Host "[4/4] セットアップウィザードを実行します..."
Write-Host ""

$engramExe = Join-Path $env:USERPROFILE ".local\bin\engram.exe"
if (-not (Test-Path $engramExe)) {
    # フォールバック: PATH から探す
    try {
        $engramExe = (Get-Command engram -ErrorAction SilentlyContinue).Source
    } catch {}
}

if (-not $engramExe) {
    Write-Host "[エラー] engram コマンドが見つかりません。"
    Write-Host "  ターミナルを再起動してから 'engram setup' を実行してください。"
    exit 1
}

& $engramExe setup

if (-not $?) {
    Write-Host ""
    Write-Host "[エラー] セットアップウィザードが失敗しました。"
    Write-Host "  問題を修正した後、'engram setup' を再実行してください。"
    exit 1
}

Write-Host ""
Write-Host "========================================"
Write-Host " インストール完了！"
Write-Host "========================================"
Write-Host ""
Write-Host "次のステップ:"
Write-Host "  - エージェント(Claude Code 等)を再起動してください"
Write-Host "  - 動作確認: engram doctor"
Write-Host ""
