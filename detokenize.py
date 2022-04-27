from sacremoses import MosesDetokenizer
# def detokenize_file(src_filename: str):
#     with open(src_filename,'r') as src:
#         lines = src.readlines()
#         return detokenize_matrix(lines,'en')
def detokenize_matrix(src_sentences, lang):
    md = MosesDetokenizer(lang=lang)
    detokenized = []
    for s in src_sentences:
        detokenized.append(md.detokenize(s.replace("@@ ", "").split(" ")))
    return detokenized
# if __name__ == '__main__':
#     with open("/cs/labs/gabis/bareluz/nematus_clean/nematus/en-de/output/non_debiased_anti_0.txt",'r') as o:
#         detokenized = detokenize_matrix(o.readlines(),"de")
#     a=1
#     # # detokenized = detokenize_file("/cs/snapless/oabend/borgr/SSMT/preprocess/data/en_ru/30.11.20/newstest2019-enru.unesc.tok.tc.bpe.en")
#     # with open("/cs/labs/gabis/bareluz/nematus_clean/nematus/en-de/output/non_debiased_anti_0.txt",'r') as o:
#     #     orig_sents = o.readlines()
#     #     for i in range(len(detokenized)):
#     #         if detokenized[i] != orig_sents[i].rstrip():
#     #             print(detokenized[i])
#     #             print(orig_sents[i])
#     #             print("*****")