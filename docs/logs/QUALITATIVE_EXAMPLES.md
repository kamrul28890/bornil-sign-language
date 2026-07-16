# Qualitative Prediction Examples

Generated from `predictions_per_example.csv` files.

Selection is automatic: exact/best, median-like representative, and
failure/pathology examples are selected by overlap scores and repetition
patterns. Human review is still required before using these in a paper.

| Track | Split | Run | Case | Row | ref/hyp tokens | chrF++ | ROUGE-L | Reference | Hypothesis | Note |
|---|---|---|---|---:|---:|---:|---:|---|---|---|
| official full-data | random-v1 | `c19d0ea08824` | exact/best | 237 | 6 / 6 | 100.00 | 1.000 | একটা অদ্ভুত সুন্দর জগত আছে ফুটবলের। | একটা অদ্ভুত সুন্দর জগত আছে ফুটবলের। | Exact match; use as a positive example. |
| official full-data | random-v1 | `c19d0ea08824` | representative | 907 | 6 / 6 | 8.25 | 0.000 | কতবার নিজের খাবার ওদের দিয়ে দিয়েছেন। | কাজের জন্য ছেলে শিশুকে পথ খুঁজেছে। | Near median chrF++; shows typical model behavior. |
| official full-data | random-v1 | `c19d0ea08824` | repetition failure | 581 | 21 / 68 | 8.76 | 0.000 | ঐ যে ভারত ভারত করে চিৎকার দিয়ে উঠে, তখন সে ভুলে যায় মায়ের রোগের কথা, সন্তানের ক্ষুধার কথা, বাবার অপমানের কথা। | তোমার বিছানায় শুয়ে থাকতে বলে নামানী বিছানা পিন মুসলিম-দুর্গ, পারলাম, পারলাম, পারলাম, পারলাম, পারলাম, পারলাম, পারলাম... | Repeated-token runaway; useful for decode-stability discussion. |
| official full-data | unseen-sentence-v1 | `60320f4c6a60` | best-overlap | 951 | 3 / 4 | 78.68 | 0.857 | হ্যাঁ, তাই না? | হ্যাঁ, না, তাই না? | Highest automatic overlap; needs human semantic check. |
| official full-data | unseen-sentence-v1 | `60320f4c6a60` | representative | 1512 | 2 / 4 | 8.04 | 0.000 | নাম্বার, প্লিজ? | আচ্ছা, আমিই শুরু করি। | Near median chrF++; shows typical model behavior. |
| official full-data | unseen-sentence-v1 | `60320f4c6a60` | repetition failure | 331 | 9 / 129 | 0.65 | 0.000 | জাতীয় সংসদ নির্বাচনে তিনি এবার মনোনয়ন পাওয়ারও চেষ্টা করেছিলেন | যখন তখন ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝম ঝ... | Repeated-token runaway; useful for decode-stability discussion. |
| official full-data | signerindep-v1 | `46a706570712` | exact/best | 840 | 4 / 4 | 100.00 | 1.000 | নিজেকে কষ্ট দিও না। | নিজেকে কষ্ট দিও না। | Exact match; use as a positive example. |
| official full-data | signerindep-v1 | `46a706570712` | representative | 267 | 9 / 6 | 7.72 | 0.000 | পাড়ামহল্লায় শিশুকিশোর ও তরুণতরুণীদের এ বিষয়ে সতর্ক করা অবশ্যকর্তব্য | বস্তুত মাধ্যমিকে এখন মেয়েদের সংখ্যাই বেশি | Near median chrF++; shows typical model behavior. |
| official full-data | signerindep-v1 | `46a706570712` | repetition failure | 965 | 6 / 255 | 0.91 | 0.000 | নদীর পাড় সাদা কাশফুলে ভরে যায়। | কাক কা কা কা কা কা কা কা কা কা কা কা কা কা কা কা কা কা কা কা কা কা কা কা কা কা কা কা কা কা কা কা কা কা কা কা কা কা কা... | Repeated-token runaway; useful for decode-stability discussion. |
| LTE12 diagnostic | random-v1 | `bc38907bece6` | exact/best | 193 | 6 / 6 | 100.00 | 1.000 | বাংলা নতুন বছরের প্রথম দিন বৈশাখ। | বাংলা নতুন বছরের প্রথম দিন বৈশাখ। | Exact match; use as a positive example. |
| LTE12 diagnostic | random-v1 | `bc38907bece6` | representative | 570 | 8 / 6 | 8.45 | 0.000 | শ্রেণিতে কারও পড়া শিখতে একটু বেশি সময় লাগে। | উপযুক্ত শব্দ দিয়ে বাক্যটি সম্পূর্ণ কর। | Near median chrF++; shows typical model behavior. |
| LTE12 diagnostic | random-v1 | `bc38907bece6` | repetition failure | 1395 | 6 / 56 | 0.93 | 0.000 | তার কথার মাঝেই কথা বলবো না। | কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে কে... | Repeated-token runaway; useful for decode-stability discussion. |
| LTE12 diagnostic | unseen-sentence-v1 | `58269a7b1003` | best-overlap | 972 | 3 / 3 | 77.01 | 0.667 | কাল দেখা হবে! | কাল দেখা হবে। | Highest automatic overlap; needs human semantic check. |
| LTE12 diagnostic | unseen-sentence-v1 | `58269a7b1003` | representative | 851 | 4 / 4 | 7.99 | 0.000 | ওকে, কী জানতে চাও? | কীসের এত ভয় তোমার? | Near median chrF++; shows typical model behavior. |
| LTE12 diagnostic | unseen-sentence-v1 | `58269a7b1003` | repetition failure | 1268 | 9 / 128 | 1.17 | 0.000 | অনেকের কাছে গল্প শুনেছি, রকেট স্টিমারে চড়ার মজাই আলাদা। | মানুষে মানুষে মানুষে মানুষে মানুষে মানুষে মানুষে মানুষে মানুষে মানুষে মানুষে মানুষে মানুষে মানুষে মানুষে মানুষে মানুষ... | Repeated-token runaway; useful for decode-stability discussion. |
| LTE12 diagnostic | signerindep-v1 | `2ff734ae1aae` | best-overlap | 396 | 7 / 7 | 67.25 | 0.857 | তার বড় বোন মার্চ মাসে জন্মগ্রহণ করে। | তার বড় বোন মার্চ মাসে মৃত্যুবরণ করে। | Highest automatic overlap; needs human semantic check. |
| LTE12 diagnostic | signerindep-v1 | `2ff734ae1aae` | representative | 975 | 10 / 10 | 7.79 | 0.000 | শিশুসাহিত্যিক সুকুমার রায় ৩০ শে অক্টোবর ১৮৮৭ খ্রিষ্টাব্দে জন্মগ্রহণ করেন। | কিন্তু তখনই শত্রুর পুতে একটু ঘুর মাথায় রাখতে বলা যায়। | Near median chrF++; shows typical model behavior. |
| LTE12 diagnostic | signerindep-v1 | `2ff734ae1aae` | repetition failure | 346 | 5 / 47 | 1.32 | 0.000 | একবার তাকাতে হয় নিজের দিকে। | প্রথম প্রথম প্রথম প্রথম প্রথম প্রথম প্রথম প্রথম প্রথম প্রথম প্রথম প্রথম প্রথম প্রথম প্রথম প্রথম প্রথম প্রথম প্রথম প্র... | Repeated-token runaway; useful for decode-stability discussion. |
