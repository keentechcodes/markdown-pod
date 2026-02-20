# OCR Benchmark Report

Results directory: `/workspace/results`

---

## Document: BATES_sample_306-324

### Timing Comparison

| Tool | Status | Total Time | Model Load | Conversion | Chars | Words |
|------|--------|-----------|------------|------------|-------|-------|
| marker-pdf | OK | 23.24s | 2.23s | 21.02s | 31,770 | 4,111 |
| nougat | OK | 223.61s | 16.38s | 207.23s | 21,194 | 3,082 |
| deepseek-ocr | OK | 526.39s | 14.83s | 511.56s | 29,424 | 4,050 |
| paddleocr | OK | 180.04s | 4.82s | 175.22s | 28,026 | 3,943 |
| docstrange | OK | 775.24s | 14.46s | 760.78s | 53,357 | 8,144 |

### Structure Detection

| Tool | Headings | Tables | Lists | Code | Math | Bold | Links |
|------|----------|--------|-------|------|------|------|-------|
| marker-pdf | 53 | 7 | 94 | 0 | 0 | 80 | 25 |
| nougat | 41 | 0 | 131 | 0 | 0 | 13 | 1 |
| deepseek-ocr | 62 | 0 | 1 | 0 | 0 | 0 | 24 |
| paddleocr | 19 | 0 | 0 | 0 | 0 | 0 | 0 |
| docstrange | 34 | 0 | 84 | 0 | 1 | 72 | 21 |

### Output Previews (first 500 chars)

#### marker-pdf

```
# Recording Your Findings

### **Recording the Anus, Rectum, and Prostate Examination**

"No perirectal lesions or fissures. External sphincter tone intact. Rectal vault without masses. Prostate smooth and nontender with palpable median sulcus. (Or in a female, uterine cervix nontender.) Stool brown and Hemoccult negative."

### **OR**

"Perirectal area inflamed; no ulcerations, warts, or discharge. Cannot examine external sphincter, rectal vault, or prostate because of spasm of external sphinct
```

#### nougat

```
## Recording Your Findings

### Recording the Anus, Rectum, and Prostate Examination

"No perirectal lesions or fissures. External sphincter tone intact. Rectal vault without masses. Prostate smooth and nontender with palpable median sulcus. (Or in a female, uterine cervix nontender.) Stool brown and Hemoccult negative."

**OR**

"Perirectal area inflamed; no ulcerations, warts, or discharge. Cannot examine external sphincter, rectal vault, or prostate because of spasm of external sphincter and 
```

#### deepseek-ocr

```
## Recording Your Findings  

## Recording the Anus, Rectum, and Prostate Examination  

"No perirectal lesions or fissures. External sphincter tone intact. Rectal vault without masses. Prostate smooth and nontender with palpable median sulcus. (Or in a female, uterine cervix nontender.) Stool brown and Hemoccult negative."  

## OR  

"Perirectal area inflamed; no ulcerations, warts, or discharge. Cannot examine external sphincter, rectal vault, or prostate because of spasm of external sphincte
```

#### paddleocr

```
## Page 1

Chapter 15| The Anus, Rectum, and Prostate
289
Recording Your Findings
RecordingtheAnus,Rectum,and
Prostate Examination
"No perirectal lesions or fissures.External sphincter tone intact.Rectal vault
without masses. Prostate smooth and nontender with palpable median sulcus.
(Or in a female, uterine cervix nontender.) Stool brown and Hemoccult negative."
OR
"Perirectal area inflamed;no ulcerations,warts,or discharge.Cannot examine
external sphincter rectal vault,or prostate because of s
```

#### docstrange

```

## Page 1

Chapter 15 | The Anus, Rectum, and Prostate <page_number>289</page_number>

# Recording Your Findings

## Recording the Anus, Rectum, and Prostate Examination

*“No perirectal lesions or fissures. External sphincter tone intact. Rectal vault without masses. Prostate smooth and nontender with palpable median sulcus. (Or in a female, uterine cervix nontender.) Stool brown and Hemoccult negative.”

OR

*“Perirectal area inflamed; no ulcerations, warts, or discharge. Cannot examine exter
```

