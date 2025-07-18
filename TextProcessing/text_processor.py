import torch
import spacy
from flair.data import Sentence as FlairSentence
from flair.models import SequenceTagger
from transformers import pipeline, AutoTokenizer, AutoModel, AutoModelForTokenClassification
from keybert import KeyBERT
from stop_words import get_stop_words
from sklearn.feature_extraction.text import TfidfVectorizer
import textstat
from collections import Counter
import numpy as np
import math
import re


class TextProcessorServer:
    def __init__(self):
        """
        Inicializa todos los modelos, pipelines y recursos necesarios para el procesamiento de texto.
        """
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.device_index = 0 if torch.cuda.is_available() else -1

        # Modelos y recursos lingüísticos
        self.nlp = spacy.load("es_core_news_md")
        self.ner_tagger = SequenceTagger.load('flair/ner-spanish-large')
        self.sentiment = pipeline(
            'sentiment-analysis',
            model='nlptown/bert-base-multilingual-uncased-sentiment',
            tokenizer='nlptown/bert-base-multilingual-uncased-sentiment',
            truncation=True,
            max_length=512,
            device=self.device_index
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            'dccuchile/bert-base-spanish-wwm-cased', use_fast=True
        )
        self.bert_model = AutoModel.from_pretrained(
            'dccuchile/bert-base-spanish-wwm-cased'
        ).to(self.device)
        self.kb_model = KeyBERT(model="paraphrase-multilingual-MiniLM-L12-v2")
        self.stop_words = get_stop_words('spanish')
        self.tfidf = TfidfVectorizer(
            max_features=100, stop_words=self.stop_words)
        self.bias_tokenizer = AutoTokenizer.from_pretrained(
            "PlanTL-GOB-ES/roberta-base-bne")
        self.bias_model = AutoModelForTokenClassification.from_pretrained(
            "PlanTL-GOB-ES/roberta-base-bne"
        ).to(self.device)
        self.bias_nlp = pipeline(
            "token-classification",
            model=self.bias_model,
            tokenizer=self.bias_tokenizer,
            aggregation_strategy="simple",
            device=self.device_index
        )
        textstat.set_lang("es")
        print(f"[Servidor] Usando dispositivo: {self.device}")

    def process_batch(self, batch_texts):
        """
        Procesa un lote de textos y obtiene estadísticas, embeddings, tópicos, entidades, sentimientos y legibilidad.
        Args:
            batch_texts (list[str]): Lista de textos a procesar.
        Returns:
            list[dict]: Lista de diccionarios con los resultados de cada texto.
        """
        # Preprocesamiento básico y extracción de POS, tokens, lemas y frases
        docs, sentences_list, tokens_list, pos_tags_list, lemmas_list = self._extract_linguistic_features(
            batch_texts)
        stats, pos_flats = self._compute_basic_stats(
            batch_texts, docs, sentences_list, tokens_list, pos_tags_list, lemmas_list)

        # NER usando Flair (procesamiento por lotes)
        flair_sentences = [FlairSentence(text) for text in batch_texts]
        self.ner_tagger.predict(flair_sentences)

        # Análisis de sentimiento a nivel de texto (transformers)
        sentiment_results = self._batch_sentiment_analysis(batch_texts)

        # Embeddings BERT a nivel de texto (CLS token)
        bert_embeddings = self._get_bert_embeddings(batch_texts)

        results = []
        for i, stat in enumerate(stats):
            # Extraer estadísticas y características intermedias para este texto
            text, sentences, tokens, lemmas, filtered_lemmas = stat["text"], stat[
                "sentences"], stat["tokens"], stat["lemmas"], stat["filtered_lemmas"]
            total_words, total_sentences, total_paragraphs = stat[
                "total_words"], stat["total_sentences"], stat["total_paragraphs"]
            total_characters, total_mayus, total_interrogations, total_exclamations = stat[
                "total_characters"], stat["total_mayus"], stat["total_interrogations"], stat["total_exclamations"]
            total_stopwords, unique_lemmatized_words, log1p = stat[
                "total_stopwords"], stat["unique_lemmatized_words"], stat["log1p"]
            pos_flat = pos_flats[i]

            # 1) Estadísticas de entidades nombradas (NER) con Flair
            ner_flat, top_entities = self._compute_ner_stats(
                flair_sentences[i], total_words, total_sentences, log1p)

            # 2) Embeddings BERT de las 5 entidades más frecuentes
            ent_emb_flat = self._get_top_entities_embeddings(top_entities)

            # 3) TF-IDF de las entidades principales
            ent_tfidf_flat = self._get_entities_tfidf(top_entities)

            # 4) Sentimiento global del texto
            sentiment_flat = self._flatten_sentiment(sentiment_results[i])

            # 5) Sentimiento por frase
            sentiment_phr_flat = self._sentiment_by_sentence(sentences)

            # 6) Extracción de tópicos principales vía TF-IDF sobre lemas
            topics, stats_flat = self._extract_topics(
                filtered_lemmas, text, total_words, log1p)

            # 7) Embeddings BERT de los 3 tópicos principales
            topic_emb_flat = self._get_top_topics_embeddings(topics)

            # 8) TF-IDF de los 3 tópicos principales
            topics_tfid_flat = self._get_topics_tfidf(topics)

            # 9) Sentimiento por tópico (top-5)
            topic_sent_flat = self._sentiment_by_topic(topics, sentences)

            # 10) Topic awareness (entropía)
            topic_awareness = self._topic_awareness(text)

            # 11) Métricas de legibilidad
            read_flat = self._compute_readability(text)

            # 12) Construcción del diccionario final con todas las métricas y estadísticas
            result = self._combine_results(
                stat, pos_flat, ent_emb_flat, ent_tfidf_flat, ner_flat, sentiment_flat,
                sentiment_phr_flat, stats_flat, topic_emb_flat, topics_tfid_flat,
                topic_sent_flat, topic_awareness, read_flat
            )
            results.append(result)
        return results

    # ======================= MÉTODOS AUXILIARES =======================

    def _extract_linguistic_features(self, batch_texts):
        """
        Extrae objetos spaCy, frases, tokens, etiquetas POS y lemas para cada texto.
        """
        docs = [self.nlp(text) for text in batch_texts]
        sentences_list = [[sent.text for sent in doc.sents] for doc in docs]
        tokens_list = [[token.text for token in doc] for doc in docs]
        pos_tags_list = [[(token.text, token.pos_)
                          for token in doc] for doc in docs]
        lemmas_list = [[token.lemma_.lower() for token in doc] for doc in docs]
        return docs, sentences_list, tokens_list, pos_tags_list, lemmas_list

    def _compute_basic_stats(self, batch_texts, docs, sentences_list, tokens_list, pos_tags_list, lemmas_list):
        """
        Calcula estadísticas básicas (palabras, frases, caracteres, etc.) y POS.
        """
        stats, pos_flats = [], []
        for i, (doc, sentences, tokens, pos_tags, lemmas) in enumerate(zip(docs, sentences_list, tokens_list, pos_tags_list, lemmas_list)):
            text = batch_texts[i]
            total_words = len(tokens)
            total_sentences = len(sentences)
            total_paragraphs = len(text.strip().split('\n\n'))
            total_characters = len(text)
            total_mayus = sum(1 for c in text if c.isupper())
            total_interrogations = text.count('?') + text.count('¿')
            total_exclamations = text.count('!') + text.count('¡')
            total_stopwords = sum(
                1 for w in tokens if w.lower() in self.stop_words)
            filtered_lemmas = [
                l for l in lemmas if l.isalpha() and l not in self.stop_words]
            unique_lemmatized_words = len(set(filtered_lemmas))
            def log1p(x): return math.log(x + 1)
            # Estadísticas de etiquetas POS
            pos_stats = {}
            for tag in set([upos for _, upos in pos_tags]):
                c = sum(1 for _, upos in pos_tags if upos == tag)
                pos_stats[tag] = {
                    'count': c,
                    'ratio_words': c / total_words if total_words else 0.0,
                    'ratio_sentences': c / total_sentences if total_sentences else 0.0
                }
            pos_flat = {}
            for tag, m in pos_stats.items():
                key = tag.replace('-', '_')
                pos_flat[f'count_{key}'] = m['count']
                pos_flat[f'ratio_words_{key}'] = m['ratio_words']
                pos_flat[f'ratio_sentences_{key}'] = m['ratio_sentences']
                pos_flat[f'count_{key}_log'] = log1p(m['count'])
            stats.append({
                "tokens": tokens,
                "sentences": sentences,
                "total_words": total_words,
                "total_sentences": total_sentences,
                "total_paragraphs": total_paragraphs,
                "total_characters": total_characters,
                "total_mayus": total_mayus,
                "total_interrogations": total_interrogations,
                "total_exclamations": total_exclamations,
                "total_stopwords": total_stopwords,
                "filtered_lemmas": filtered_lemmas,
                "unique_lemmatized_words": unique_lemmatized_words,
                "lemmas": lemmas,
                "text": text,
                "log1p": log1p
            })
            pos_flats.append(pos_flat)
        return stats, pos_flats

    def _batch_sentiment_analysis(self, batch_texts):
        """
        Realiza análisis de sentimiento a nivel de texto usando transformers.
        Si ocurre un error, procesa uno a uno como fallback.
        """
        try:
            return self.sentiment(batch_texts)
        except Exception:
            return [self.sentiment(t)[0] for t in batch_texts]

    def _get_bert_embeddings(self, batch_texts):
        """
        Obtiene los embeddings BERT (CLS) para cada texto del batch.
        """
        encodings = self.tokenizer(batch_texts, return_tensors='pt', padding=True,
                                   truncation=True, max_length=self.tokenizer.model_max_length)
        encodings = {k: v.to(self.device) for k, v in encodings.items()}
        with torch.no_grad():
            bert_outputs = self.bert_model(**encodings)
        return bert_outputs.last_hidden_state[:, 0, :].cpu().numpy()

    def _compute_ner_stats(self, flair_sentence, total_words, total_sentences, log1p):
        """
        Calcula estadísticas sobre entidades nombradas reconocidas por Flair.
        """
        ents = flair_sentence.get_spans('ner')
        ner_stats = {}
        for tag in set([e.tag for e in ents]):
            c = sum(1 for e in ents if e.tag == tag)
            ner_stats[tag] = {
                'count': c,
                'ratio_words': c / total_words if total_words else 0.0,
                'ratio_sentences': c / total_sentences if total_sentences else 0.0
            }
        ner_flat = {}
        for tag, m in ner_stats.items():
            key = tag.replace('-', '_')
            ner_flat[f'ner_count_{key}'] = m['count']
            ner_flat[f'ner_ratio_words_{key}'] = m['ratio_words']
            ner_flat[f'ner_ratio_sentences_{key}'] = m['ratio_sentences']
            ner_flat[f'ner_count_{key}_log'] = log1p(m['count'])
        top_entities = [e.text for e in ents]
        return ner_flat, top_entities

    def _get_top_entities_embeddings(self, top_entities):
        """
        Obtiene los embeddings BERT de las 5 entidades más frecuentes.
        """
        ent_embs = []
        hidden_size = self.bert_model.config.hidden_size
        counts_ent = Counter(top_entities)
        top_entities = [ent for ent, _ in counts_ent.most_common(5)]
        for ent in top_entities:
            enc = self.tokenizer(
                ent,
                return_tensors='pt',
                truncation=True,
                max_length=self.tokenizer.model_max_length
            ).to(self.device)
            with torch.no_grad():
                out = self.bert_model(**enc)
            emb = out.last_hidden_state[0, 0, :].detach().cpu().numpy()
            ent_embs.append(emb)
        if not ent_embs:
            ent_flat_arr = np.zeros(5 * hidden_size)
        else:
            ent_flat_arr = np.concatenate(
                ent_embs + [np.zeros(hidden_size)] * (5 - len(ent_embs))
            )[:5 * hidden_size]
        return {
            f'emb_ent{i+1}_{j}': float(ent_flat_arr[i * hidden_size + j])
            for i in range(5) for j in range(hidden_size)
        }

    def _get_entities_tfidf(self, top_entities):
        """
        Calcula el TF-IDF de las 5 entidades principales.
        """
        ent_tfidf_arr = np.zeros(5)
        if top_entities:
            vec_ent = TfidfVectorizer()
            tfidf_vec = vec_ent.fit_transform(top_entities).toarray().flatten()
            ent_tfidf_arr[: min(len(tfidf_vec), 5)] = tfidf_vec[:5]
        return {
            f'ent_tfidf_{i+1}': float(ent_tfidf_arr[i]) for i in range(5)
        }

    def _flatten_sentiment(self, sentiment):
        """
        Estructura el resultado de sentimiento a nivel de texto.
        """
        return {
            'sentiment_label': sentiment['label'].split(' ')[0],
            'sentiment_score': sentiment['score']
        }

    def _sentiment_by_sentence(self, sentences):
        """
        Calcula el sentimiento para cada frase y la proporción de positivas, negativas y neutrales.
        """
        pos_count = neg_count = neu_count = 0
        for sentt in sentences:
            res = self.sentiment(sentt)
            lab = res[0]['label'].lower()
            if 'pos' in lab:
                pos_count += 1
            elif 'neg' in lab:
                neg_count += 1
            else:
                neu_count += 1
        total_phr = len(sentences)
        return {
            'positive_phrases_count': pos_count,
            'negative_phrases_count': neg_count,
            'neutral_phrases_count': neu_count,
            'positive_phrases_ratio': pos_count / total_phr if total_phr else 0.0,
            'negative_phrases_ratio': neg_count / total_phr if total_phr else 0.0,
            'neutral_phrases_ratio': neu_count / total_phr if total_phr else 0.0
        }

    def _extract_topics(self, filtered_lemmas, text, total_words, log1p):
        """
        Extrae los 5 tópicos principales vía TF-IDF sobre lemas y cuenta su frecuencia en el texto.
        """
        text_lem = " ".join(filtered_lemmas)
        vec2 = self.tfidf.fit_transform([text_lem])
        feats = self.tfidf.get_feature_names_out()
        scores2 = np.asarray(vec2.sum(axis=0)).ravel()
        idx2 = np.argsort(scores2)[-5:][::-1]
        topics = [(feats[i], float(scores2[i])) for i in idx2]
        stats_flat = {}
        for j, (topic, sc) in enumerate(topics, start=1):
            cnt = len(re.findall(
                rf"\b{re.escape(topic)}\b", text, flags=re.IGNORECASE))
            stats_flat[f'topic_{j}_count'] = cnt
            stats_flat[f'topic_{j}_ratio_words'] = cnt / \
                total_words if total_words else 0.0
            stats_flat[f'topic_{j}_score'] = sc
            stats_flat[f'topic_{j}_count_log'] = log1p(cnt)
        for j in range(len(topics) + 1, 6):
            stats_flat[f'topic_{j}_count'] = 0.0
            stats_flat[f'topic_{j}_ratio_words'] = 0.0
            stats_flat[f'topic_{j}_score'] = 0.0
            stats_flat[f'topic_{j}_count_log'] = 0.0
        return topics, stats_flat

    def _get_top_topics_embeddings(self, topics):
        """
        Obtiene los embeddings BERT de los 3 tópicos principales.
        """
        top3 = [t for t, _ in topics[:3]]
        emb_list = []
        hidden_size = self.bert_model.config.hidden_size
        for tpc in top3:
            enc2 = self.tokenizer(
                tpc,
                return_tensors='pt',
                truncation=True,
                max_length=self.tokenizer.model_max_length
            ).to(self.device)
            with torch.no_grad():
                out2 = self.bert_model(**enc2)
            emb_list.append(
                out2.last_hidden_state[0, 0, :].detach().cpu().numpy())
        if not emb_list:
            emb_arr = np.zeros(3 * hidden_size)
        else:
            emb_arr = np.concatenate(
                emb_list + [np.zeros(hidden_size)] * (3 - len(emb_list)))[:3 * hidden_size]
        return {
            f'topic_emb{j+1}_{k}': float(emb_arr[j * hidden_size + k])
            for j in range(3) for k in range(hidden_size)
        }

    def _get_topics_tfidf(self, topics):
        """
        Obtiene el TF-IDF de los 3 tópicos principales.
        """
        tfidf3 = np.zeros(3)
        if topics:
            names = [t for t, _ in topics[:3]]
            vec3 = TfidfVectorizer(max_features=3)
            arr3 = vec3.fit_transform(names).toarray().flatten()
            n = min(len(arr3), 3)
            tfidf3[: n] = arr3[:n]
        return {f'topic_tfidf_{j+1}': float(tfidf3[j]) for j in range(3)}

    def _sentiment_by_topic(self, topics, sentences):
        """
        Calcula el sentimiento de fragmentos relacionados con cada tópico principal.
        """
        topic_sent_list = []
        for topic, _ in topics[:5]:
            related = [s for s in sentences if topic in s]
            if related:
                frag3 = " ".join(related)
                res3 = self.sentiment(frag3)[0]
                topic_sent_list.append(
                    {'label': res3['label'], 'score': res3['score']})
            else:
                topic_sent_list.append({'label': '3 stars', 'score': 0.0})
        topic_sent_flat = {}
        for j, ts in enumerate(topic_sent_list, start=1):
            topic_sent_flat[f'topic_{j}_sentiment_label'] = ts['label'].split(' ')[
                0]
            topic_sent_flat[f'topic_{j}_sentiment_score'] = ts['score']
        for j in range(len(topic_sent_list) + 1, 6):
            topic_sent_flat[f'topic_{j}_sentiment_label'] = ''
            topic_sent_flat[f'topic_{j}_sentiment_score'] = 0.0
        return topic_sent_flat

    def _topic_awareness(self, text):
        """
        Calcula la entropía de tópicos (topic awareness) sobre el texto.
        """
        vec4 = self.tfidf.transform([text]).toarray().ravel()
        p = vec4 / vec4.sum() if vec4.sum() else vec4
        entropy = -np.sum([pi * np.log(pi) for pi in p if pi > 0])
        return 1 / (1 + entropy) if entropy else 0.0

    def _compute_readability(self, text):
        """
        Calcula métricas de legibilidad para textos en español.
        """
        return {
            'Fernández-Huerta-readability': textstat.fernandez_huerta(text),
            'Szigriszt-Pazos-readability': textstat.szigriszt_pazos(text),
            'Gutiérrez-Polini-readability': textstat.gutierrez_polini(text)
        }

    def _combine_results(self, stat, pos_flat, ent_emb_flat, ent_tfidf_flat, ner_flat, sentiment_flat,
                         sentiment_phr_flat, stats_flat, topic_emb_flat, topics_tfid_flat,
                         topic_sent_flat, topic_awareness, read_flat):
        """
        Unifica todas las métricas y resultados en un diccionario final por texto.
        """
        total_words = stat["total_words"]
        total_sentences = stat["total_sentences"]
        total_paragraphs = stat["total_paragraphs"]
        total_characters = stat["total_characters"]
        total_mayus = stat["total_mayus"]
        total_interrogations = stat["total_interrogations"]
        total_exclamations = stat["total_exclamations"]
        total_stopwords = stat["total_stopwords"]
        unique_lemmatized_words = stat["unique_lemmatized_words"]
        log1p = stat["log1p"]

        return {
            'total_words': total_words,
            'total_words_log': log1p(total_words),
            'total_sentences': total_sentences,
            'total_sentences_log': log1p(total_sentences),
            'total_paragraphs': total_paragraphs,
            'total_paragraphs_log': log1p(total_paragraphs),
            'total_characters': total_characters,
            'total_characters_log': log1p(total_characters),
            'total_mayus': total_mayus,
            'total_mayus_log': log1p(total_mayus),
            'total_interrogations': total_interrogations,
            'total_interrogations_log': log1p(total_interrogations),
            'total_exclamations': total_exclamations,
            'total_exclamations_log': log1p(total_exclamations),
            'char_per_word_ratio': total_characters / total_words if total_words else 0.0,
            'char_per_sentence_ratio': total_characters / total_sentences if total_sentences else 0.0,
            'word_per_sentence_ratio': total_words / total_sentences if total_sentences else 0.0,
            'char_per_paragraph_ratio': total_characters / total_paragraphs if total_paragraphs else 0.0,
            'sentences_per_paragraph_ratio': total_sentences / total_paragraphs if total_paragraphs else 0.0,
            'word_per_paragraph_ratio': total_words / total_paragraphs if total_paragraphs else 0.0,
            'total_stopwords': total_stopwords,
            'total_stopwords_log': log1p(total_stopwords),
            'unique_lemmatized_words': unique_lemmatized_words,
            'unique_lemmatized_words_log': log1p(unique_lemmatized_words),
            'unique_lemmatized_words_ratio': unique_lemmatized_words / total_words if total_words else 0.0,
            **pos_flat,
            **ent_emb_flat,
            **ent_tfidf_flat,
            **ner_flat,
            **sentiment_flat,
            **sentiment_phr_flat,
            **stats_flat,
            **topic_emb_flat,
            **topics_tfid_flat,
            **topic_sent_flat,
            'topic_awareness': topic_awareness,
            **read_flat
        }
