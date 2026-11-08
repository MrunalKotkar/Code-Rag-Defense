Subject: Weekly update (Nov 9) - coding-docs track, Step 1 started

Dear Prof. Stamp,

This week I started Step 1 of the coding-docs track: building the document
index and the question splits. The goal for Sunday, Nov 15 is a checked mGTE
search index of the CodeRAG-Bench documents plus a fresh slice of DevDocs pages
and PyPI READMEs, the 80/20 proxy/test split, the target list with fake
package names, and the real row counts, chunk count, embedding time and
NDCG@10, which will replace the estimates in my plan.

A few questions where I would value your decision, ideally at Wednesday office
hours:

1. BigCodeBench. ImportSnare took its Python questions from BigCodeBench, so I
   would like to add its 1,140 tasks next to DS-1000 and ODEX. Is that fine?
2. Stack Overflow overlap. DS-1000 and ODEX were built from Stack Overflow, so
   some posts in the corpus nearly copy a test question. I will flag them
   (MinHash similarity, plus exact matches for ODEX, whose ids are Stack
   Overflow question ids) and send you the list. Should flagged posts be
   dropped from the corpus?
3. Running the unit tests. I plan to run generated code on a separate Colab
   CPU runtime with no Drive and no API keys, instead of Docker. Is that
   acceptable?
4. Cost. Paid models and extra Colab units come to about $40-$85 on top of my
   Colab Pro plan. Is that acceptable, and is there funding for it?
5. Near-duplicate questions in the split. DS-1000's 1,000 tasks come from 453
   original problems (the rest are rewordings), and ODEX has several phrasings
   of the same Stack Overflow question. I plan to keep each family on one side
   of the 80/20 split, so the attacker never sees a rewording of a test
   question. Do you agree?

Best regards,
Mrunal
