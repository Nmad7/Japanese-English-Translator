import PySimpleGUI as sg
import pickle
import nagisa
# import numpy as np
import unicodedata
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam


class ai_translator:
    def __init__(self):
        self.model = load_model('model.hdf5')
        self.model.compile(loss = 'sparse_categorical_crossentropy',optimizer = Adam(),metrics = ['accuracy'])
        with open('ja_tokenizer.pickle', 'rb') as handle:
            self.ja_tokenizer = pickle.load(handle)
        with open('eng_tokenizer.pickle', 'rb') as handle:
            self.eng_tokenizer = pickle.load(handle)

    def convert_to_seq(self,text):
        spaced_text = " ".join(word for word in nagisa.tagging(text).words)
        spaced_text = ''.join(ascii_text for ascii_text in unicodedata.normalize('NFKD', spaced_text))
        spaced_text = "startl " + spaced_text + " endl"
        text_seq = self.ja_tokenizer.texts_to_sequences([spaced_text])
        # pad sequences with 0 values
        text_seq = pad_sequences(text_seq, maxlen=34, padding='post')
        return text_seq

    def translate_sequence(self,seq_text):
        preds = self.model.predict_classes(seq_text)
        # preds = np.argmax(self.model.predict(seq_text), axis=-1)
        return preds[0]

    def convert_to_text(self,pred):
        def get_word(n, tokenizer):
            for word, index in tokenizer.word_index.items():
                if index == n:
                    return word
            return None

        temp = []
        for j in range(len(pred)):
            t = get_word(pred[j], self.eng_tokenizer)
            if j > 0:
                if (t == get_word(pred[j - 1], self.eng_tokenizer)) or (t == None):
                    temp.append('')
                else:
                    temp.append(t)
            else:
                if (t == None):
                    temp.append('')
                else:
                    temp.append(t)

        return ' '.join(temp)

    def translate(self,text):
        seq_text = self.convert_to_seq(text)
        seq_translated = self.translate_sequence(seq_text)
        translated_text = self.convert_to_text(seq_translated)
        # remove startl and endl
        removal_list = ["startl", "endl"]
        edit_string_as_list = translated_text.split()
        final_list = [word for word in edit_string_as_list if word not in removal_list]
        final_string = ' '.join(final_list)
        return final_string

def Window():
    sg.theme('Reddit')
    layout = [[(sg.Text('Please enter Japanese text to translate', size=[40, 1]))],
              [sg.Multiline(size=(70, 5), enter_submits=True),
               sg.Button('SEND', button_color=(sg.YELLOWS[0], sg.BLUES[0]))],
              [(sg.Text('Translated text', size=[40, 1]))],
              [sg.Multiline(size=(80, 10),key='OUT'),
               sg.Button('EXIT', button_color=(sg.YELLOWS[0], sg.BLUES[0]))]]

    window = sg.Window('Japanese to English', layout, default_element_size=(30, 2))

    trans = ai_translator()
    while True:
        event, value = window.read()
        if event == 'SEND':
            translated_sen = trans.translate(value[0].strip())
            window['OUT'].update(translated_sen)
        else:
            break
    window.close()

Window()