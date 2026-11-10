# OCR research and scanned-PDF handling

StudyRAG is designed to run locally on modest laptops (including 8 GB RAM) without a paid inference service. It already uses Tesseract for Bengali + English scanned PDFs.

## What we checked

- Source: [baidu/Unlimited-OCR](https://github.com/baidu/Unlimited-OCR) and [baidu/Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2).
- Baidu's MIT-licensed 3B-parameter vision-language OCR model supports advanced parsing across many pages.
- Its published Transformers inference is tested on an **NVIDIA A800 80 GB GPU**, with CUDA and a custom model implementation. The recommended SGLang serving path is also specialized for GPU use.
- The official materials do not establish fast, accurate Bengali scanned-book OCR on an 8 GB CPU-only laptop. We did not benchmark Baidu OCR2 for StudyRAG.

**Decision:** keep lightweight Tesseract with `ben+eng` as the default. Adding Baidu OCR2 as a mandatory dependency would make the app much heavier without evidence that it improves the user's local workflow.

## Improvement adopted

Some scanned PDFs have a tiny selectable text layer containing just a page number, watermark or header, while the page's main text is in a full-page raster image. Previously StudyRAG skipped OCR whenever `page.get_text()` returned any text, even if it was only `12`.

The updated loader:

1. Uses normal PDF text extraction first.
2. If extracted text is under 80 characters, checks image bounding boxes without rendering the page.
3. If one image covers at least 65% of the page, runs the existing Tesseract OCR.
4. Uses the OCR result when it contains more text than the sparse extracted text; otherwise keeps the existing text.
5. Preserves the PDF page number and temporary/session storage workflow.

**Why the heuristic:** small logos, signatures and diagrams should not trigger OCR when the page is otherwise textual. It remains heuristic: odd layouts and scans spread over several smaller images may be missed.

The OCR cache version is now `v2` to ensure PDFs whose old cached result contained only a page number get reevaluated. The first reupload of a previously indexed scanned PDF may take longer; later uploads reuse the new cache. Delete `~/.cache/studyrag/ocr/` to clear extracted text, or set `STUDYRAG_OCR_CACHE_DIR` to a private directory.

## Test coverage

`tests/test_ocr_mixed_pages.py` exercises real PyMuPDF PDFs containing a full-page image and selectable page number, a small logo on a text page, a large background on a readable page, and an OCR result shorter than the existing text. Tesseract calls are mocked to keep CI offline.

Run:

```bash
python -m ruff check .
python -m pytest -q
```

For actual Bengali OCR performance, check local PDFs against their original pages. Synthetic CI tests confirm page-routing behavior, **not** recognition accuracy.

## Upstream references

- https://github.com/baidu/Unlimited-OCR
- https://huggingface.co/baidu/Unlimited-OCR-2

No Baidu code or model weights have been copied into StudyRAG.
