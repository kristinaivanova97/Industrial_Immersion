# GEC
This project was created as a result of summer working off in Asmanov's neural nets company and still under development.

This is a project for detection and correction some general mistakes in Russian writing language.
For now it could with greate accuracy correct tsya/tisya and with reasonable accuracy - n/nn misspellings.

The base of our model is BertForTokenClassification from Transformers library https://huggingface.co/transformers/master/model_doc/bert.html. 
Tutorial for evaluating BERT tokenization and the model http://mccormickml.com/2019/07/22/BERT-fine-tuning/. 
Thanks to https://github.com/yuanxiaosc/BERT-for-Sequence-Labeling-and-Text-Classification/blob/master/run_sequence_labeling.py open project for some ideas for perfoming appropriate tagging.
Model was compared with similar one from deep-pavlov library, but the last have not showed any improvements in all the metrics.

For now this model is implemented in https://orfo.ashmanov.org for testing. 

### Project description

For training and testing all tokens should be represented with their indices (input_ids, input_mask and label_ids)
These matrices (of size num_of_sentences*512) are saved into hdf5 files for simplicity and memory  saving.

For launching scripts you will need to put appropriate paths&values to configuration files.
They are named templates_*.json here.

To test our model from cmd type: python TestingModel.py -f <file_name.txt> 
or change in template_config_stand.json parameter "write_from_terminal" to true value and launch with python TestingModel.py then type num of sentences and afterwards - your sentences.

There is an ability to retrain model on extra sets.

For simplicity, the model is wrapped into class in Class_for_execution.py, which could be used for testing

### Future work

- Retrain model (on checked sentences from wikipedia + sentences with "длина") + get better metrics
- Add more mistake types and finetune model

