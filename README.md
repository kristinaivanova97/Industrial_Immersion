# Industrial_Immersion
This project was created as a result of summer working off in Asmanov's neural nets company.

This is a project for detection and correction some general mistakes in Russian writing language.
For now it could with greate accuracy correct tsya/tisya and n/nn misspellings.

The base of our model is BertForTokenClassification from Transformers library https://huggingface.co/transformers/master/model_doc/bert.html. Tutorial for evaluating BERT tokenization and the model http://mccormickml.com/2019/07/22/BERT-fine-tuning/. Thanks to https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification/blob/master/run_sequence_labeling.py open project for some ideas for perfoming appropriate tagging.

For now this model will be implemented in https://orfo.ashmanov.org for testing. 
