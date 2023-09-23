# Instructions
This is a temporary repository for evaluating VQA on OpenFlamingo by random selecting ICEs(RS), retrieving ICEs by image-level and question-level
similarity(SIIR and SQQR).

## dataset
vqaV2: https://visualqa.org/download.html

## model
OpenFlamingo-9B: https://huggingface.co/openflamingo/OpenFlamingo-9B-vitl-mpt7b
(language model: mosaicml/mpt-7b)

Specific installation can be found in README.md.

## evaluate
1. follow the instructions in README.md to install the model and 
create the environment.
   
2. download the retrieval result file:
https://pan.quark.cn/s/049360520c0e
   
3.run the bash file: open_flamingo/scripts/run_eval_vqa.sh

** some paths need to be revised in the files.

4. the output files we need:

4.1 the intermediate results (3 retrieval types * 5 shot modes)
   
the format will be: vqav2results_RS_4.json

4.2 a final result file: openflamingo_v2_vqaV2_RS_SIIR_SQQR.json