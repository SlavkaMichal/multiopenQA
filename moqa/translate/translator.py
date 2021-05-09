import os
from typing import Union, List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from argparse import Namespace

import requests
import uuid

TRANSLATORS = {
    'opus-mt-mul-en'  : ['ca', 'es', 'os', 'eo', 'ro', 'fy', 'cy', 'is', 'lb', 'su', 'an', 'sq', 'fr', 'ht', 'rm', 'cv',
                         'ig', 'am', 'eu', 'tr', 'ps', 'af', 'ny', 'ch', 'uk', 'sl', 'lt', 'tk', 'sg', 'ar', 'lg', 'bg',
                         'be', 'ka', 'gd', 'ja', 'si', 'br', 'mh', 'km', 'th', 'ty', 'rw', 'te', 'mk', 'or', 'wo', 'kl',
                         'mr', 'ru', 'yo', 'hu', 'fo', 'zh', 'ti', 'co', 'ee', 'oc', 'sn', 'mt', 'ts', 'pl', 'gl', 'nb',
                         'bn', 'tt', 'bo', 'lo', 'id', 'gn', 'nv', 'hy', 'kn', 'to', 'io', 'so', 'vi', 'da', 'fj', 'gv',
                         'sm', 'nl', 'mi', 'pt', 'hi', 'se', 'as', 'ta', 'et', 'kw', 'ga', 'sv', 'ln', 'na', 'mn', 'gu',
                         'wa', 'lv', 'jv', 'el', 'my', 'ba', 'it', 'hr', 'ur', 'ce', 'nn', 'fi', 'mg', 'rn', 'xh', 'ab',
                         'de', 'cs', 'he', 'zu', 'yi', 'ml', 'mul', 'en'],
    'tgt_constituents': {
        'sjn_Latn',
        'cat',
        'nan',
        'spa', 'ile_Latn', 'pap', 'mwl', 'uzb_Latn', 'mww', 'hil', 'lij', 'avk_Latn', 'lad_Latn', 'lat_Latn',
        'bos_Latn', 'oss', 'epo', 'ron', 'fry', 'cym', 'toi_Latn', 'awa', 'swg', 'zsm_Latn', 'zho_Hant', 'gcf_Latn',
        'uzb_Cyrl', 'isl', 'lfn_Latn', 'shs_Latn', 'nov_Latn', 'bho', 'ltz', 'lzh', 'kur_Latn', 'sun', 'arg',
        'pes_Thaa', 'sqi', 'uig_Arab', 'csb_Latn', 'fra', 'hat', 'liv_Latn', 'non_Latn', 'sco', 'cmn_Hans', 'pnb',
        'roh', 'chv', 'ibo', 'bul_Latn', 'amh', 'lfn_Cyrl', 'eus', 'fkv_Latn', 'tur', 'pus', 'afr', 'brx_Latn', 'nya',
        'acm', 'ota_Latn', 'cha', 'ukr', 'xal', 'slv', 'lit', 'zho_Hans', 'tmw_Latn', 'kjh', 'ota_Arab', 'war', 'tuk',
        'sag', 'myv', 'hsb', 'lzh_Hans', 'ara', 'tly_Latn', 'lug', 'brx', 'bul', 'bel', 'vol_Latn', 'kat', 'gan',
        'got_Goth', 'vro', 'ext', 'afh_Latn', 'gla', 'jpn', 'udm', 'mai', 'ary', 'sin', 'tvl', 'hif_Latn', 'cjy_Hant',
        'bre', 'ceb', 'mah', 'nob_Hebr', 'crh_Latn', 'prg_Latn', 'khm', 'ang_Latn', 'tha', 'tah', 'tzl', 'aln', 'kin',
        'tel', 'ady', 'mkd', 'ori', 'wol', 'aze_Latn', 'jbo', 'niu', 'kal', 'mar', 'vie_Hani', 'arz', 'yue', 'kha',
        'san_Deva', 'jbo_Latn', 'gos', 'hau_Latn', 'rus', 'quc', 'cmn', 'yor', 'hun', 'uig_Cyrl', 'fao', 'mnw', 'zho',
        'orv_Cyrl', 'iba', 'bel_Latn', 'tir', 'afb', 'crh', 'mic', 'cos', 'swh', 'sah', 'krl', 'ewe', 'apc', 'zza',
        'chr', 'grc_Grek', 'tpw_Latn', 'oci', 'mfe', 'sna', 'kir_Cyrl', 'tat_Latn', 'gom', 'ido_Latn', 'sgs', 'pau',
        'tgk_Cyrl', 'nog', 'mlt', 'pdc', 'tso', 'srp_Cyrl', 'pol', 'ast', 'glg', 'pms', 'fuc', 'nob', 'qya', 'ben',
        'tat', 'kab', 'min', 'srp_Latn', 'wuu', 'dtp', 'jbo_Cyrl', 'tet', 'bod', 'yue_Hans', 'zlm_Latn', 'lao', 'ind',
        'grn', 'nav', 'kaz_Cyrl', 'rom', 'hye', 'kan', 'ton', 'ido', 'mhr', 'scn', 'som', 'rif_Latn', 'vie', 'enm_Latn',
        'lmo', 'npi', 'pes', 'dan', 'fij', 'ina_Latn', 'cjy_Hans', 'jdt_Cyrl', 'gsw', 'glv', 'khm_Latn', 'smo', 'umb',
        'sma', 'gil', 'nld', 'snd_Arab', 'arq', 'mri', 'kur_Arab', 'por', 'hin', 'shy_Latn', 'sme', 'rap', 'tyv', 'dsb',
        'moh', 'asm', 'lad', 'yue_Hant', 'kpv', 'tam', 'est', 'frm_Latn', 'hoc_Latn', 'bam_Latn', 'kek_Latn', 'ksh',
        'tlh_Latn', 'ltg', 'pan_Guru', 'hnj_Latn', 'cor', 'gle', 'swe', 'lin', 'qya_Latn', 'kum', 'mad', 'cmn_Hant',
        'fuv', 'nau', 'mon', 'akl_Latn', 'guj', 'kaz_Latn', 'wln', 'tuk_Latn', 'jav_Java', 'lav', 'jav', 'ell', 'frr',
        'mya', 'bak', 'rue', 'ita', 'hrv', 'izh', 'ilo', 'dws_Latn', 'urd', 'stq', 'tat_Arab', 'haw', 'che', 'pag',
        'nno', 'fin', 'mlg', 'ppl_Latn', 'run', 'xho', 'abk', 'deu', 'hoc', 'lkt', 'lld_Latn', 'tzl_Latn', 'mdf',
        'ike_Latn', 'ces', 'ldn_Latn', 'egl', 'heb', 'vec', 'zul', 'max_Latn', 'pes_Latn', 'yid', 'mal', 'nds'}
    }


class Translator:
    def __init__(self, device='cpu', norwegian_azure=False):
        self.device = device
        self.norwegian_azure = norwegian_azure
        self.supported = ["ar", "da", "de", "es", "en", "fi", "fr", "hu", "it",
                          "ja", "ko", "nl", "no", "pl", "pt", "ru", "sv", "th", "tr", ]
        self.keys2opus = {'ar': 'ara', 'da': 'dan', 'de': 'deu', 'es': 'spa', 'fi': 'fin',
                          'fr': 'fra', 'hu': 'hun', 'it': 'ita', 'ja': 'jav', 'nl': 'nld',
                          'pl': 'pol', 'pt': 'por', 'ru': 'rus', 'sv': 'swe', 'th': 'tha', 'tr': 'tur'}
        self.opus_mul_langs = [lang for lang in self.keys2opus]
        self.opus_translators = ['en-mul', 'mul-en',
                                 'de-no', 'no-de', 'ko-en']

        # Azure translator setup
        self.key_var_name = 'AZURE_KEY'
        if not self.key_var_name in os.environ:
            self.azure_key = None
        else:
            self.azure_key = os.environ[self.key_var_name]

        endpoint = 'https://api.cognitive.microsofttranslator.com/'
        path = '/translate'
        self.url = endpoint + path
        self.headers = {
            'Ocp-Apim-Subscription-Key': self.azure_key,
            'Content-type'             : 'application/json',
            'X-ClientTraceId'          : str(uuid.uuid4())
            }

        # translators
        self.translators = {}

    def __call__(self, text, src_lang, dst_langs):
        if src_lang == 'no':
            pass
        if src_lang == 'ko':
            pass
        else:
            return self.translate_opus_mul(text, src_lang, dst_langs)

    def init_translator(self, translator):
        model_name = 'Helsinki-NLP/opus-mt-' + translator
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        prepare_seq2seq_batch = tokenizer.prepare_seq2seq_batch
        decode = tokenizer.decode

        return Namespace(model=model,
                         generate=model.generate,
                         model_name=model_name,
                         tokenizer=tokenizer,
                         prepare_seq2seq_batch=prepare_seq2seq_batch,
                         decode=decode)

    def del_translators(self):
        self.translators.clear()

    def translate(self, translator: str, text: List[str]) -> List[str]:
        if translator not in self.translators:
            self.translators[translator] = self.init_translator(translator)

        try:
            with torch.no_grad():
                input = self.translators[translator].tokenizer(text, return_tensors='pt', max_length=512,
                                                               truncation=True).to(self.device)
                output = self.translators[translator].generate(**input)
            translations = [self.translators[translator].decode(indeces, skip_special_tokens=True) for indeces in
                            output]
        except Exception as e:
            raise e
        return translations

    def translate_opus_mul(self, text: Union[List[str], str], src_lang, dst_langs: Union[str, List[str]]) -> Dict[
        str, str]:
        if dst_langs == 'all':
            dst_langs = self.supported.copy()
        elif dst_langs == 'opus-mul':
            dst_langs = self.opus_mul_langs.copy()
            dst_langs.append('en')
        elif type(dst_langs) == list:
            # TODO test if dst_langs is subset of opus_mul_langs
            dst_langs = dst_langs.copy()  # must be a copy otherwise it will change dst_langs array
        else:
            raise ValueError(f"Unsupported value dst_langs={dst_langs}")

        if src_lang not in self.opus_mul_langs + ['en']:
            raise NotImplementedError(f"Language {src_lang} is not supported.")
        # if set(dst_langs).intersection(set(dst_langs)):
        #     raise NotImplementedError(f"Cannot translate to: {set(dst_langs).intersection(set(dst_langs))}")

        translations = {}
        if src_lang != 'en':
            text_en = self.to_en(text, src_lang)
            if src_lang in dst_langs:
                translations[src_lang] = text
                dst_langs.remove(src_lang)
        else:
            text_en = text
        if 'en' in dst_langs:
            translations['en'] = text_en
            dst_langs.remove('en')
        tmp = self.from_en(text_en, dst_langs)
        translations.update(tmp)

        return translations

    def to_en(self, text: List[str], src_lang):
        text_en = self.translate('mul-en', text)[0]
        return text_en

    def from_en(self, text_en, dst_langs: List[str]):
        inp_text = []
        dst_langs = dst_langs.copy()
        if 'en' in dst_langs:
            dst_langs.remove('en')
        for lang in dst_langs:
            inp_text.append(f">>{self.keys2opus[lang]}<< {text_en}")

        text_mul = self.translate('en-mul', inp_text)
        translations = {'en': text_en}
        for lang, translation in zip(dst_langs, text_mul):
            translations[lang] = translation

        return translations

    def from_no_to_de(self, text, dst_langs=None):
        if 'no-de' not in self.translators:
            self.translators['no-de'] = self.init_translator('no-de')
        src_en = self.translators['no-de'](text)
        return src_en

    def from_de_to_no(self, text):
        if 'de-no' not in self.translators:
            self.translators['de-no'] = self.init_translator('de-no')
        src_en = self.translators['de-no'](text)
        return src_en

    def from_ko_to_en(self, text):
        if 'ko-en' not in self.translators:
            self.translators['ko-en'] = self.init_translator('ko-en')

        src_en = self.translators['ko-en'](text)
        return src_en

    def to_no(self, text, src_lang):
        if self.norwegian_azure:
            return self.translate_azure(text, dst_lang='nb', src_lang=src_lang)
        else:
            src_en = self.to_en(text, src_lang)
            src_de = self.from_en(src_en, dst_langs=['de'])
            return self.from_de_to_no(src_de)

    def to_ko(self, text, src_lang):
        return self.translate_azure(text, dst_lang='ko', src_lang=src_lang)

    def translate_azure(self, text, dst_lang, src_lang):
        if self.azure_key == None:
            raise RuntimeError('Please set/export the environment variable: {}'.format(self.key_var_name))
        if type(dst_lang) == list:
            params = {
                'api-version': '3.0',
                'from'       : 'en',
                'to'         : dst_lang
                }
        else:
            params = {
                'api-version': '3.0',
                'from'       : 'en',
                'to'         : [dst_lang]
                }
        body = [{'text': text} for text in text]

        request = requests.post(self.url, params=params, headers=self.headers, json=body)
        response = request.json()[0]
        if 'translations' not in response:
            raise RuntimeError(f"Something went wrong with response\n"
                               f"Error: {response}\n"
                               f"params: {params}\nbody: {body}")
        return [translation['text'] for translation in response['translations']]
