from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict
from PIL import Image
import io

# OCR が使える環境ならコメントアウトを外す
try:
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

app = FastAPI(title="Toho Pachi Backend")

# CORS（フロントのHTMLから呼べるようにする）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 必要に応じて絞る
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ManualInput(BaseModel):
    gousei_prob: float   # 例: 169 → 1/169
    total_spin: float    # 総回転数
    diff: float          # 差枚／差玉（マイナスもOK）


class AnalyzeResult(BaseModel):
    score: int
    features: Dict[str, Optional[float]]
    comment: str
    raw_ocr: Optional[str] = None


def calc_score(features: Dict[str, Optional[float]]) -> (int, str):
    """
    めちゃシンプルな暫定スコア計算。
    あとで好きなロジックに差し替えてOK。
    """
    score = 0
    comments = []

    gousei = features.get("gousei_prob")
    spin = features.get("total_spin")
    diff = features.get("diff")

    # 合成確率：軽いほどプラス
    if gousei is not None:
        if gousei <= 160:
            score += 40
            comments.append(f"合成が軽め (1/{int(gousei)}) で好印象。")
        elif gousei <= 200:
            score += 20
            comments.append(f"合成はそこそこ (1/{int(gousei)})。")
        else:
            comments.append(f"合成がやや重い (1/{int(gousei)})。")

    # 回転数：回されてるほど情報量として加点
    if spin is not None:
        if spin >= 2000:
            score += 20
            comments.append(f"総回転 {int(spin)} 回転でデータ量は十分。")
        elif spin >= 1000:
            score += 10
            comments.append(f"総回転 {int(spin)} 回転でそこそこ。")
        else:
            comments.append(f"総回転 {int(spin)} 回転でまだ様子見レベル。")

    # 差枚：大きく凹んでる台を少しプラス評価（好みで調整してOK）
    if diff is not None:
        if diff <= -2000:
            score += 20
            comments.append(f"現在差枚 {int(diff)} 枚で大きく凹み。上げ狙い候補かも。")
        elif diff <= -1000:
            score += 10
            comments.append(f"現在差枚 {int(diff)} 枚でややマイナス。")
        elif diff >= 2000:
            score -= 10
            comments.append(f"現在差枚 {int(diff)} 枚でかなり出ている台。追いづらいかも。")
        else:
            comments.append(f"現在差枚 {int(diff)} 枚で極端ではない。")

    score = max(0, min(100, score))

    if not comments:
        comments.append("情報が少ないため、スコアは参考程度。")

    return score, " / ".join(comments)


@app.post("/analyze_manual", response_model=AnalyzeResult)
async def analyze_manual(data: ManualInput):
    features = {
        "gousei_prob": data.gousei_prob,
        "total_spin": data.total_spin,
        "diff": data.diff,
    }
    score, comment = calc_score(features)
    return AnalyzeResult(score=score, features=features, comment=comment)


@app.post("/analyze", response_model=AnalyzeResult)
async def analyze_image(image: UploadFile = File(...)):
    if not image.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="画像ファイルを送ってください。")

    contents = await image.read()
    raw_ocr = None
    features = {"gousei_prob": None, "total_spin": None, "diff": None}

    if OCR_AVAILABLE:
        try:
            pil_image = Image.open(io.BytesIO(contents))
            raw_ocr = pytesseract.image_to_string(pil_image, lang="jpn+eng")
            # TODO: OCRしたテキストから数値を抜き出すロジックをここに追加
        except Exception:
            raw_ocr = "OCRに失敗しました（あとでチューニング）。"

    score, comment = calc_score(features)
    return AnalyzeResult(score=score, features=features, comment=comment, raw_ocr=raw_ocr)


@app.get("/")
async def root():
    return {"message": "toho-pachi-backend running"}



