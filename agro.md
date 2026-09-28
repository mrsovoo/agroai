# AGROAI LOYIHASI AUDITI (2026-09-28)

**Loyiha yo'li:** `/opt/goinfre/plankcla/agroai` · **Rejim:** READ → ANALYZE → TEST SAFELY → REPORT (faqat o'qish va xavfsiz diagnostika)

> ⚠️ **Audit oshkorligi (koproq yozuv):** Bitta diagnostik testda yozish effekti bo'ldi. `agroai ingest-dir` ni sandboxda ishga tushirganimda, CLI loyiha ildizini cwd/HOME dan emas, **o'rnatilgan paket joylashuvidan** aniqlagani uchun `knowledge/uzbekistan/region_context_example.md` loyiha bazasiga yozib yuborildi: `data/database/agroai.db` **2 hujjat / 3 bo'lak → 3 hujjat / 5 bo'lak** bo'ldi. Yangi qatorlar 100% coverage bilan tekshirildi. Hech bir source fayl, kod yoki konfiguratsiya o'zgartirilmadi. Qatorlarni o'chirish o'zi ma'lumotni o'zgartirish bo'lardi, shuning uchun bazaga tegilmadi va bu shu yerda to'liq yozib qo'yildi.

---

## 1. Loyihaninng hozirgi holati

**Faza 1 "offline-first poydevor"** mohiyatan to'liq qurilgan kod bazasi: 88 paket fayli (~19 000 qator Python) + 25 test fayli (testlar bilan jami 23 775 qator). Runtime bog'liqliklari nol (faqat stdlib). Barcha 290 test 0.71 sekundda o'tadi. Butun pipeline (config → CLI → agent → RAG retrieval → safety → API → vision chegaralari) real, ishlaydigan kod — **faqat lokal LLM/vision/embedding runtime** atayin yo'q va "not configured" deb halol xabar beradi. Bilimlar bazasida atigi 3 ta kichik seed hujjat bor.

AI ishlashi to'xtagan joyda ish o'rtada qolmagan. Kod toza (TODO yo'q, sintaksis xatosi yo'q); qolgan narsalar — oxirgi kod qatlamida kiritilgan **3 ta kichik CLI ulash xatosi** (plans/models/datasets buyruqlari) va **mavjud bo'lmagan `.env`** (zararsiz — standart qiymatlar xavfsiz).

## 2. Loyiha strukturasi

```text
agroai/                        # loyiha ildizi
├── pyproject.toml             # paket metadata; deps=[] ; "agroai" console script
├── README.md                  # 18.9 KB, aniq va halol
├── .env.example               # to'liq konfig shabloni;  .env MAVJUD EMAS
├── .gitignore                 # .venv, data DB, models, datasets ignore qilinadi
├── docs/ARCHITECTURE.md       # roadmap→kod xaritasi, real vs chegara holatlari
├── agroai/                    # 88 Python fayl
│   ├── agent/    agent, planner, executor, memory, prompts
│   ├── api/      service.py (framework-siz HTTP API), contracts, quotas
│   ├── cli/      main.py (~880 qator, 19 subcommand), doctor.py
│   ├── config/   settings.py (env + .env parser, keshlangan singleton)
│   ├── database/ database.py (SQLite + FTS5, versions, candidates, memory)
│   ├── datasets/ versiyalangan dataset menejeri + deterministik splits
│   ├── diagnosis/ retrieval-ga asoslangan differensial tashxis dvigateli
│   ├── evaluation/ runner, suites, metrics, regression gate
│   ├── ingestion/ 8 bosqichli pipeline, pdf/html (rad etadi), structure, verification
│   ├── knowledge/ documents, metadata, sources, manager, candidates, versions, graph
│   ├── memory/    episodic + user memory (rozilik bilan)
│   ├── models/    base, loader, inference, registry + fine_tuning/
│   ├── retrieval/ retriever (SQLite FTS5), embeddings, vector_store
│   ├── safety/    guardrails, validation
│   ├── skills/    catalog (15 skill), router
│   ├── sources/   ishonchli manbalar reyestri, updater (tarmoqqa chiqmaydi)
│   ├── tools/     6 ta deterministik vosita
│   └── vision/    image/video inspektsiyasi, preprocessing, engines (model yo'q)
├── tests/                     # 25 fayl, 290 test
├── scripts/                   # export_dataset.py, reset_database.sh
├── knowledge/                 # 13 kategoriya katalogi, 3 ta seed .md fayl
├── data/                      # database/agroai.db (3 hujjat), bo'sh indexes/documents
├── models/, datasets/         # bo'sh (faqat README)
└── .venv/                     # Python 3.12.3 venv, editable install, pytest 9.1.1
```

## 3. Amalga oshirilgan komponentlar

| Komponent | Holat | Fayllar | Dalil |
| --------- | ----- | ------- | ----- |
| Loyiha konfiguratsiyasi / packaging | **COMPLETE** | `pyproject.toml` | `pip list` da editable `agroai 0.1.0`; console script ishlaydi |
| Konfiguratsiya tizimi | **COMPLETE** | `agroai/config/settings.py` | env + o'z `.env` parseri + standartlar; `doctor` to'liq konfigni ko'rsatadi; aliaslar (`AGROAI_LOCAL_MODEL`/`AGROAI_MODEL_PATH`) |
| CLI | **COMPLETE (2 xato)** | `agroai/cli/main.py`, `doctor.py` | 19 subcommand; quyida test qilingan; `plans` `--user` yo'qligidan qulaydi, `models`/`datasets` circular import dan qulaydi |
| Doctor (muhit tekshiruvi) | **COMPLETE** | `agroai/cli/doctor.py` | Exit 0; 14 OK, 0 WARNING, 7 NOT CONFIGURED, 0 ERROR |
| Logging / xato boshqaruvi | **COMPLETE** | hamma joyda | tiplan exceptionlar, ingestion har fayl xatosini hisobot qiladi |
| Bilim modellari + provenance | **COMPLETE** | `knowledge/` (9 modul) | front-matter parser, ishonch darajalari 1/2/3, sha-256 hash, sitatalar faqat saqlangan hujjatlardan |
| Ingestion pipeline | **COMPLETE (TXT/MD)** | `ingestion/pipeline.py` + boshqalar | 8 bosqich jonli: load→extract→clean→til→metadata→dedup→chunk→store; jonli ishga tushirish: 3 ko'rildi, 1 qo'shildi, 2 takror, 0 xato |
| PDF/HTML/JSON/CSV readerlar | **SKELETON (halol)** | `ingestion/pdf.py`, `html.py` | `NotImplementedError` amaliy xabar bilan ("pip install 'agroai[pdf]'"); soxta ajratib olish yo'q |
| SQLite saqlash + FTS5 | **COMPLETE** | `database/database.py` | 12 jadval, shu jumladan `chunks_fts`, `document_versions`, `knowledge_candidates`, `user_memory`; jonli so'rovlar ishlaydi |
| Leksik qidiruv | **COMPLETE** | `retrieval/retriever.py` | `agroai search "pomidor"` reytingli FTS5 natijalar; inglizcha so'rov ham topdi |
| Embeddings | **NOT IMPLEMENTED (halol chegarа)** | `retrieval/embeddings.py` | `doctor`: NOT CONFIGURED; provayder `none` |
| Vector store | **COMPLETE mexanika, SOZLANMAGAN** | `retrieval/vector_store.py` | real in-memory cosine store + indexer, testlangan; backend `none` |
| LLM interfeysi | **COMPLETE shartnoma, dvigatel YO'Q** | `models/base.py`, `inference.py` | `LocalLLMProvider` NOT CONFIGURED deb xabar beradi; soxta matn ishlab chiqarmaydi |
| Model loader/registry | **COMPLETE** | `models/loader.py`, `registry.py` | GGUF magic-byte tekshiruvi; lifecycle bosqichlari bilan reyestr |
| Fine-tuning strukturasi | **PARTIAL (atayin)** | `models/fine_tuning/` | dataset sxemasi/IO + LoRA config + run menejer real; `LoRATrainer.train()` `TrainingNotAvailable` raises; trening runtime yo'q |
| Agent (planner/executor/memory) | **COMPLETE** | `agent/` | `agroai plan` `ec_ds_m: 6.5` ni parse qilgan 5 bosqichli deterministik rejani chiqaradi |
| Skills router | **COMPLETE** | `skills/` | 15 skill; routing self-test o'tadi |
| Deterministik tashxis | **COMPLETE** | `diagnosis/engine.py` | retrieval-ga asoslangan differensial, hech qachon yakuniy xulosa emas; doctor self-test o'tadi |
| Vositalar | **COMPLETE** | `tools/` (6 ta) | calculator (AST), unit_converter, soil, irrigation, weather, veterinary — barchasi ro'yxatda |
| Safety (guardrails+validation) | **COMPLETE** | `safety/` | eval suite: dosage→high risk, manba majburiy; strict mode yoniq |
| Vision | **PARTIAL** | `vision/` | real fayl inspektsiyasi (format/resolyutsiya/hajm, preprocessing plan); tavsif modelsiz imkonsiz — halol xabar |
| Video | **SKELETON** | `vision/video.py` | faqat konteyner/metadata; frame tanlash `NotImplementedError` |
| API | **COMPLETE** | `api/service.py` | framework-siz HTTP xizmat, 13 route, kvotalar, auth; jonli `GET /api/v1/health` → 200 "degraded" (LLM yo'q) |
| Kvotalar/plans | **COMPLETE (1 CLI xato)** | `api/quotas.py` | 3 plan (free/pro/internal) har action limitlari bilan; usage store ishlaydi; `agroai plans` qulayadi |
| Telegram bot | **NOT IMPLEMENTED** | — | telegram moduli umuman yo'q |
| Admin panel | **NOT IMPLEMENTED** | — | eng yaqin ekvivalent: CLI `candidates` review oqimi + API knowledge routelari |
| Tashqi qidiruv / updater | **SKELETON (rad etadi)** | `sources/updater.py` | `not_implemented, network_used=False` qaytaradi; siyosatda 15 ishonchli host |
| Testlar | **COMPLETE** | `tests/` 25 fayl | **290 o'tdi, 0 muvaffaqiyatsiz, 0 xato, 0.71s** |
| Bilim mazmuni | **SKELETON** | `knowledge/` | faqat 3 ta seed fayl; 10/13 kategoriya bo'sh |

## 4. Qisman amalga oshirilgan komponentlar

| Komponent | Hozirgi holat | Yetishmayapti |
| --------- | ------------- | -------------- |
| LLM generatsiya | Provider shartnomasi + planner/validator ulangan; `ask` retrieval natijalarini ko'rsatadi + halol "not configured" | `InferenceEngine` ortidagi istalgan inference dvigateli (Ollama/llama.cpp/GGUF) |
| Embeddings / semantik qidiruv | Vector store + indexer real va testlangan | Embedding provayderi; vektor backend o'chiq (`none`) |
| Vision | Fayl forensikasi ishlaydi (test: PNG 4x4, 73 bayt to'g'ri parse qilindi) | Vision model; pikseldan belgi ajratish; video frame dekodlash |
| Fine-tuning | Dataset sxemasi, splits, versiyalash, run provenance — barchasi real | Haqiqiy trening runtime (PyTorch/transformers + GPU); Q&A sinteziga generator kerak |
| Bilimlar bazasi | Pipeline, dedup, verification (100% coverage) ishlaydi | Haqiqiy mazmun: 10/13 kategoriyada nol hujjat (Faza 2) |
| PDF/HTML ingestion | Aniq xato xabarlari bilan ro'yxatdan o'tgan readerlar | Ajratib olish (`pypdf` o'rnatilmagan) |
| API/Telegram/Admin | HTTP API real; admin-o'xshash oqimlar CLI da mavjud | Telegram bot; web admin UI; deploy |

## 5. Buzilgan komponentlar

| Komponent | Xato | Sabab | Fayl |
| --------- | ---- | ----- | ---- |
| `agroai models` | `ImportError: cannot import name 'DatasetManager' from partially initialized module 'agroai.datasets.manager' (circular import)` | modul darajasidagi tsikl: `datasets.manager` → `models.fine_tuning.__init__` → `fine_tuning.manager` → orqaga `datasets.manager` ga | `agroai/models/fine_tuning/manager.py:35`, `agroai/datasets/manager.py:44` |
| `agroai datasets` | xuddi shu circular import | xuddi shu tsikl | xuddi shu joylar |
| `agroai plans` | `AttributeError: 'Namespace' object has no attribute 'user'` (exit 1) | `cmd_plans` `args.user` ni o'qiydi, lekin `plans` subparserida `--user` umuman e'lon qilinmagan | `agroai/cli/main.py:560` va `:813` |

Boshqa hech narsa buzilmagan: `compileall` toza, 290/290 test o'tadi, qolgan 16 CLI buyrug'i exit 0.

## 6. CLI holati

```text
COMMAND                  STATUS
agroai                   OK   (interaktiv rejim; banner chiqadi, Ctrl-D → exit 0)
agroai --help            OK   (19 subcommand ro'yxati)
agroai version           OK   (AgroAI 0.1.0)
agroai doctor            OK   (exit 0; 14 OK / 0 WARNING / 7 NOT CONFIGURED / 0 ERROR)
agroai search            OK   ("pomidor" → 3 reytingli FTS5 natija; inglizcha ham topdi)
agroai ingest            OK   (qo'llab-quvvatlanmaydigan kengaytma → toza xato, exit 1)
agroai ingest-dir        OK   (3 ko'rildi, 1 qo'shildi, 2 takror, 0 xato — yuqoridagi ogohlantirishni ko'ring)
agroai sources           OK   (3 manba vaqt belgisi bilan)
agroai ask               OK   (KB natijalari + halol NOT CONFIGURED xabari; exit 0)
agroai plan              OK   (to'liq deterministik reja; tool bosqichi EC=6.5 ni parse qildi)
agroai tools             OK   (6 vosita)
agroai skills            OK   (15 skill; --for routing ishlaydi)
agroai candidates        OK   (bo'sh ro'yxat, halol xabar)
agroai eval              OK   (smoke suite: 7 pass, 1 halol skip, keyword_hit=1.000)
agroai verify            OK   (3/3 hujjat VERIFIED, 100% coverage)
agroai structure         OK   (3 markdown bo'lim aniqlandi)
agroai vision            OK   (fayl inspektsiyasi ishlaydi; tavsif halol tarzda mavjud emas)
agroai api               OK   (jonli health 200/degraded; --routes jadvali ishlaydi)
agroai plans             BROKEN (AttributeError: --user argumenti yo'q)
agroai models            BROKEN (circular import)
agroai datasets          BROKEN (circular import)
```

## 7. Bilim pipeline holati

```text
[COMPLETE]   Documents        (3 ta seed fayl; kategoriya papkadan avtomatik)
   ↓
[COMPLETE]   Parser           (TXT/MD; PDF/HTML/JSON/CSV ro'yxatda, halol rad etadi)
   ↓
[COMPLETE]   Cleaning         (qatorlar, bo'sh joylar, boshqaruv belgilari)
   ↓
[COMPLETE]   Chunking         (size 800 / overlap 120, spanlar tekshirilgan)
   ↓
[COMPLETE]   Metadata         (front matter, ishonch darajasi, sha-256, til uz/ru/en)
   ↓
[NOT IMPL.]  Embeddings       (provayder none — halol chegarа)
   ↓
[NOT IMPL.]  Vector DB        (backend none; in-memory store bor, sozlanmagan)
   ↓
[COMPLETE]   Retrieval        (SQLite FTS5 leksik, BM25, LIKE fallback)
```

## 8. AI/model holati

- **Kodlash yordamchisi:** tegishli emas — repoda AI-assist xususiyati yo'q.
- **Tashqi LLM interfeysi:** `models/base.py` da `LLMProvider` shartnomasi (`generate`/`chat`/`health_check`); `models/inference.py` da `LocalLLMProvider` — u faqat `InferenceEngine` orqali matn chiqaradi, aks holda NOT CONFIGURED deb xabar beradi. Bulut provider kodi umuman yo'q — core tarmoqqa ulanmaydi.
- **Lokal model qo'llab-quvvatlashi:** `models/loader.py` GGUF fayllarini haqiqatan tekshiradi (magic bytes, hajm, oila) va `models/` katalogini skanerlaydi; `doctor` natijani ko'rsatadi. Dvigatel ulanmagan — `get_inference_engine` kelajakdagi ulash nuqtasi.
- **Kelajakdagi AgroAI modeli:** roadmap (README + ARCHITECTURE.md) `Ochiq/Baza Model + agro dataset → fine-tune → AgroAI Model` ni ko'zda turibdi; faktlar retrievable KB da qoladi (model og'irligiga yozilmaydi). Kod: quyidagi strukturalardan boshqasi yo'q.
- **Trening/fine-tuning infratuzilmasi:** dataset sxemasi + JSONL IO + deterministik splits + versiyalangan `DatasetManager` + `ModelRegistry` + `FineTuningManager` (provenance bilan run planlash/yozib borish) — barchasi real; `LoRATrainer.train()` va `build_instruction_dataset()` atayin tushuntirish bilan rad etadi. `scripts/export_dataset.py` extractive, manbaga bog'langan yozuvlarni eksport qiladi.

## 9. Agent holati

- **Planner:** COMPLETE — rule-based, deterministik, ko'rish mumkin (`agroai plan` barcha 5 bosqichni argumentlari bilan ko'rsatadi).
- **Executor:** COMPLETE — bosqichlarni ishga tushiradi, har birining statusini yozadi, muvaffaqiyatsizlikda grazt degradatsiya qiladi.
- **Vositalar:** COMPLETE — 6 ta deterministik vosita; faqat barcha parametrlar mavjud bo'lganda rejalashtiriladi.
- **Xotira:** COMPLETE — in-process cheklangan suhbat xotirasi (`agent/memory.py`), plus rozilik bilan yoqiladigan `user_memory` jadvali va `memory/` modullari.
- **Muhokama (reasoning):** rule-based rejalashtirish + skills routing (15 skill) + retrieval-ga asoslangan differensial tashxis dvigateli; dvigatel ulanmaguncha LLM muhokamasi yo'q.
- **Manba tekshiruvi:** COMPLETE — provenance ingestda, javobdan keyingi validatsiya (o'ylab topilgan URLlar, manbasiz miqdorlar, haddan tashqari ishonch, grounding coverage ≥0.25), bilim nomzodlari faqat odam tekshiruvidan keyin qabul qilinadi.

## 10. Vision/video holati

**Mavjud:** media turini aniqlash, o'lcham/hajm/davomiylik parse qilish, preprocessing rejalashtirish (1024px byudjet, "resize shart emas" test qilingan), fayl faktlari + foydalanuvchi matnli tavsifi + KB retrieval + tashxis dvigatelini birlashtirgan `MediaAnalyzer`, cheklovlar chiqishda ochiq ko'rsatiladi. **Yetishmayapti:** haqiqiy rasm tushunish (vision model sozlanmagan), video dekodlash/frame tanlash (`NotImplementedError`). Holat dizaynga mos: model paydo bo'lguncha faqat inspektsiya.

## 11. API / Telegram / Admin holati

- **API:** Faza 1 uchun COMPLETE — framework-siz HTTP xizmat, 13 hujjatlashtirilgan route (`/api/v1/chat|ask|search|vision|knowledge…|models|plans|health|version`), server tomonida kvota nazorati (free/pro/internal planlar), API-kalit qo'llab-quvvatlashi (`AGROAI_API_KEYS`, hozir bo'sh → lokal rejim). Jonli health: HTTP 200, `status: degraded` — faqat LLM yo'qligi uchun.
- **Telegram:** NOT IMPLEMENTED — bot kodi, dependency, konfig yo'q.
- **Admin panel:** UI sifatida NOT IMPLEMENTED — ekvivalent imkoniyatlar CLI da (`candidates --review/--approve/--reject`, `verify`, `eval`) va API knowledge routelarida mavjud.

## 12. Testlar

- **Jami: 290** (25 fayl, `tests/`).
- **O'tdi: 290 · Muvaffaqiyatsiz: 0 · Xato: 0 · Skip: 0** (`290 passed in 0.71s`).
- Eslatma: README da "237 test" deyilgan — hozirgi 290 ga nisbatan eskirgan (README oxirgi kod qo'shilishlaridan orqada: API, skills, diagnosis, datasets, vision qatlamlari).
- Testlanmagan joylar (test yo'q, muvaffaqiyatsizlik emas): `agroai models`/`datasets`/`plans` CLI uchun end-to-end exit-kod testlari yo'q (ular faqat boshqacha import tartibida/argument yo'qligida qulaydi), API HTTP-darajasidagi integratsiya testlari yo'q, Telegram uchun testga hech narsa yo'q.

## 13. Bog'liqliklar (dependencies)

**O'rnatilgan** (`.venv/bin/pip list`): `agroai 0.1.0` (editable), `pytest 9.1.1`, pytest o'z dep-lari (`iniconfig`, `packaging`, `pluggy`, `Pygments`), `pip 26.2.1`.

- **Runtime bog'liqliklar:** yo'q — core atayin faqat stdlib va ishlaydi.
- **Yetishmayapti (ixtiyoriy extras, o'rnatilmagan):** `pypdf` (`agroai[pdf]` — real PDF ajratish uchun), `numpy` (`agroai[embeddings]` — embeddings bosqichi uchun). Hozir ishlayotgan hech narsa uchun kerak emas.
- **Ishlatilmagan/buzilgan:** yo'q. Versiya ziddiyatlari yo'q.

## 14. Git holati

```text
git status / git log:  "fatal: not a git repository"
```

**Bu loyiha Git repozitoriy emas** — versiya tarixi yo'q, oxirgi commitlarni yoki oxirgi o'zgarish vaqtini aniqlashning imkoni yo'q. Yagona dalil — fayl mtime lar: paket source lari oxirgi marta **28-sentabr, 20:20–21:02** da yozilgan (datasets, docs, skills, api, vision, models qatlamlari), `tests/` — **21:34** da. Ya'ni eng yangi kod va testlar shu auditdan bir necha soat oldingi sessiyadan.

## 15. Oxirgi ishlagan nuqta

- **OXIRGI MA'LUM ISHLAGAN HOLAT (jonli tekshirilgan):** Python 3.12.3 venv, editable install; 290/290 test; CLI, RAG retrieval, ingestion, vositalar, safety, API, doctor, evaluation — barchasi ishlaydi; bazada 2 ta ingest qilingan seed hujjat (2026-09-27 09:35 UTC da ingest qilingan).
- **ISH QAYERDA TO'XTAGAN:** katta kengaytirish o'tishidan keyin (taxminiy tartib, qatlam bo'yicha): skills router → diagnosis engine → knowledge candidates → dataset versiyalash → model registry → API/kvota qatlami → vision qatlami → ularning testlari. To'xtash nuqtasining dalili: **eng yangi uch CLI buyrug'i (`models`, `datasets`, `plans`) testlar ushlab olmaydigan tarzda buzgan** (modul darajasidagi circular import; argparse argumenti yo'q) — sessiya yangi CLI buyruqlarini oxirigacha ishga tushirib ko'rmasdan tugagan. `knowledge/uzbekistan/region_context_example.md` seed (mtime 28-sentabr 20:24) ilgari ingest qilinganlardan (27-sentabr) keyin yaratilgan va bugungacha hech qachon ingest qilinmagan edi.

## 16. Hozirgi muammoning ildiz sababi

"AI ishlamay qoldi" degan gap CLI ishlamasligiga ishora bo'lsa, aniq sabablar:

1. **Circular import** — `agroai/datasets/manager.py:44` modul darajasida `agroai.models.fine_tuning.dataset` dan `DatasetRecord, read_jsonl, write_jsonl` import qiladi, bu `agroai/models/fine_tuning/__init__.py` ni ishga tushiradi, u `agroai/models/fine_tuning/manager.py:35` dan `FineTuningManager` ni import qiladi, u esa hali ishga tushayotgan `agroai.datasets.manager` dan orqaga `DatasetManager` ni import qiladi. Avval `agroai.datasets` import qilinsa → qulash; avval `agroai.models.fine_tuning` import qilinsa → ishlaydi. Shuning uchun `agroai models` va `agroai datasets` exit 1, lekin unit testlari o'tadi (boshqacha import tartibi).
2. **Argparse nomuvofiqligi** — `cmd_plans` (`main.py:560`) `args.user` ni o'qiydi, lekin `plans` subparserida (`main.py:813`) `--user` e'lon qilinmagan; `agroai plans` har doim `AttributeError` beradi (exit 1). Latent xato ham bor: `_create_dataset` (`args.min_chars` va boshqalar) o'qiydigan argumentlar subparserda e'lon qilinmagan.
3. **Ikkilamchi (bug emas, dizayn):** `.env` yo'q va lokal LLM sozlanmagan → `ask` halol "not configured" deb javob beradi; **Git repo yo'q** → qulash oynasi hech qachon tarix/CI tomonidan ushlanmagan.

Qiziq fakt: audit davomida ingest qilingan `region_context_example.md` ingestion ishlaganini va tinjqoq turganini isbotlaydi — u shunchaki o'sha seed fayl yaratilgandan beri ishga tushirilmagan edi.

## 17. Nima allaqachon yaxshi (qayta yozmaslik kerak)

- Butun **agent dvigateli** (planner/executor/prompts) — deterministik, ko'rish mumkin, testlangan.
- **Ingestion pipeline + provenance/ishonch metadata** — haqiqatan yaxshi qurilgan, 100% coverage bilan tekshirilgan.
- **SQLite + FTS5 saqlash qatlami** va **leksik retrieval**.
- **Safety qatlamlari** (guardrails + post-validation) — loyihaning ajralib turuvchi qismi.
- **"Hech narsa soxta emas" chegara patterni** (hamma joyda halol `NotImplementedError`/NOT CONFIGURED) — izchil va qimmatli; saqlash kerak.
- **API xizmati + kvotalar**, **skills katalog/router**, **tashxis dvigateli**, **regression gate bilan evaluation framework**.
- **Konfiguratsiya tizimi** va **doctor**.
- **Test to'plami** (290 test, tez, oflayn) — har qanday tuzatish uchun xavfsizlik to'ri.

## 18. Avval nimani tuzatish kerak

- **P0 — loyihani to'sib qo'ygan:**
  1. `datasets ↔ models.fine_tuning` circular import ini uzish (`agroai models` + `agroai datasets` ochiladi).
  2. `plans` subparseriga yetishmayotgan `--user` argumentini qo'shish (`agroai plans` ochiladi); shu yerda dataset yaratish argumentlarini ham e'lon qilish.
  3. Git ni ishga tushirish va boshlang'ich commit — loyihada hozir nol versiya nazorati bor.
- **P1 — muhim:**
  4. Har bir subcommand ni end-to-end ishga tushirib exit kodlarini tekshiradigan CLI smoke-test qo'shish (ikkala bugni ham ushlab olgan bo'lardi); README dagi eskirgan test sonini yangilash (237→290).
- **P2 — keyinroq:**
  5. Haqiqiy Faza-2 bilim kontentini ingest qilish (10/13 kategoriya bo'sh); PDF kerak bo'lganda `pypdf` o'rnatish.
  6. `get_inference_engine` ortiga `InferenceEngine` ulash (Ollama/llama.cpp) — `ask` "not configured" dan javob generatsiyasiga o'tadi.
- **P3 — kelajak:**
  7. Embeddinglar + vektor backend (mexanika tayyor); Telegram bot; admin UI; trening runtime; avtomatik ishonchli manba yangilanishlari.

## 19. Tavsiya etilgan keyingi qadam

**`agroai models` / `agroai datasets` circular import ini va `agroai plans` dagi yetishmayotgan `--user` argumentini tuzatish, so'ng har bir subcommand exit 0 chiqarishini tasdiqlovchi CLI smoke-test qo'shish.** Bu to'liq CLI yuzasini tiklaydigan eng kichik o'zgarish, yuqoridagi dalillarga to'liq asoslangan va hech narsani qayta yozmasdan aynan uchta buzilgan kirish nuqtasiga tegadi.

---

*Butun diagnostika xavfsiz/o'qish-rejimida o'tkazildi, yuqorida ochiq bayon etilgan DB ingestdan (boshidagi ogohlantirish) tashqari. Hech narsa o'rnatilmadi, o'chirilmadi, refaktor qilinmadi, commit qilinmadi; boshqa loyihalarga tegilmadi.*
