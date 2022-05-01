from typing import Any, Dict, List, Optional, Union, cast
from overrides import overrides

import os
import random
import pickle
import logging

import numpy as np
import pyarrow as pa

from dataclasses import dataclass

from functools import lru_cache

from torch._C import dtype

from allennlp.data.fields import Field, TextField, LabelField, MetadataField, ArrayField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, PretrainedTransformerTokenizer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers.token_class import Token

from datasets import load_dataset
from datasets.features import Features, Value, ClassLabel, Sequence
from datasets.arrow_dataset import Dataset
from datasets import set_caching_enabled

logger = logging.getLogger(__name__)

set_caching_enabled(os.environ.get('HF_DISABLE_CACHE', '').strip() != "1")


@dataclass
class SentencePairFeature:
    key1: str
    key2: str


@DatasetReader.register("huggingface")
class HuggingFaceDatasetReader(DatasetReader):
    def __init__(
        self,
        *,
        path: Optional[str] = None,
        name: Optional[str] = None,
        tokenizer: Tokenizer,
        token_indexers: Dict[str, TokenIndexer] = None,
        combine_input_fields: Optional[bool] = None,
        ensure_whitespace_between: bool = False,
        combine_opposite: bool = False,
        config_kwargs: Dict[str, Any] = None,
        **kwargs,
    ) -> None:
        super().__init__(manual_distributed_sharding=False, **kwargs)  # Right now disabled

        self.class_mappings = {'cooking': {0: 'cooking_query', 1: 'cooking_recipe'}, 'transport': {0: 'transport_query', 1: 'transport_taxi', 2: 'transport_ticket', 3: 'transport_traffic'}, 'email': {0: 'email_addcontact', 1: 'email_query', 2: 'email_querycontact', 3: 'email_sendemail'}, 'general': {0: 'general_affirm', 1: 'general_commandstop', 2: 'general_confirm', 3: 'general_dontcare', 4: 'general_explain', 5: 'general_greet', 6: 'general_joke', 7: 'general_negate', 8: 'general_praise', 9: 'general_quirky', 10: 'general_repeat'}, 'qa': {0: 'qa_currency', 1: 'qa_definition', 2: 'qa_factoid', 3: 'qa_maths', 4: 'qa_stock'}, 'recommendation': {0: 'recommendation_events', 1: 'recommendation_locations', 2: 'recommendation_movies'}, 'audio': {0: 'audio_volume_down', 1: 'audio_volume_mute', 2: 'audio_volume_other', 3: 'audio_volume_up'}, 'alarm': {0: 'alarm_query', 1: 'alarm_remove', 2: 'alarm_set'}, 'social': {0: 'social_post', 1: 'social_query'}, 'datetime': {0: 'datetime_convert', 1: 'datetime_query'}, 'iot': {0: 'iot_cleaning', 1: 'iot_coffee', 2: 'iot_hue_lightchange', 3: 'iot_hue_lightdim', 4: 'iot_hue_lightoff', 5: 'iot_hue_lighton', 6: 'iot_hue_lightup', 7: 'iot_wemo_off', 8: 'iot_wemo_on'}, 'lists': {0: 'lists_createoradd', 1: 'lists_query', 2: 'lists_remove'}, 'calendar': {0: 'calendar_query', 1: 'calendar_remove', 2: 'calendar_set'}, 'music': {0: 'music_dislikeness', 1: 'music_likeness', 2: 'music_query', 3: 'music_settings'}, 'play': {0: 'play_audiobook', 1: 'play_game', 2: 'play_music', 3: 'play_podcasts', 4: 'play_radio'}, 'takeaway': {0: 'takeaway_order', 1: 'takeaway_query'}}
        self.scenario = path.split('/')[-1]
        print(self.scenario)
        self._tokenizer = tokenizer
        if isinstance(self._tokenizer, PretrainedTransformerTokenizer):
            assert not self._tokenizer._add_special_tokens
            # TODO what if no combining is required?
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        if combine_input_fields is not None:
            self._combine_input_fields = combine_input_fields
        else:
            self._combine_input_fields = isinstance(self._tokenizer, PretrainedTransformerTokenizer)

        self._ensure_whitespace_between = ensure_whitespace_between
        if ensure_whitespace_between:
            assert self._combine_input_fields

        self.path = path
        self.name = name

        if config_kwargs is None:
            config_kwargs = {}
        self.config_kwargs = config_kwargs


        self._keys_to_combine = [
            ("sentence1", "sentence2"),
            ("premise", "hypothesis"),
            ("question", "sentence"),
            ("question1", "question2"),
            ("question", "passage"),
        ]
        self._combine_opposite = combine_opposite
        if combine_opposite:
            self._keys_to_combine = [
                (second, first)
                for first, second
                in self._keys_to_combine
            ]

    @lru_cache(maxsize=None)
    def get_huggingface_dataset(self, path: str, name: str = None, *, split: str = None) -> Dataset:
        if split is not None:
            return self.get_huggingface_dataset(path, name)[split]
        logger.info(f"Loading {path}/{name}...")
        return load_dataset(path, name, **self.config_kwargs)

    @overrides
    def _read(self, file_path: str):
        # TODO what if file_path is list or dict??

        if ":" in file_path:
            paths: List[str] = file_path.split(":")
            for path in paths:
                yield from self._read(path)
            return

        name, _, split = file_path.rpartition("/")

        if self.name is not None:
            if name:
                # assert not name, "Dataset Configuration already provided"
                logger.warning("Dataset Configuration already provided")
                logger.warning("Overriding the base configuration...")
            else:
                name = self.name

        dataset: Dataset = self.get_huggingface_dataset(self.path, name, split=split)
        features: Features = dataset.features.copy()

        yield from self._read_from_huggingface_dataset(dataset, features)

    def _read_from_huggingface_dataset(
        self,
        dataset: Dataset,
        features: Features
    ):

        if self._combine_input_fields:
            for key1, key2 in self._keys_to_combine:
                sentence1 = features.get(key1)
                sentence2 = features.get(key2)
                if not isinstance(sentence1, Value) or not isinstance(sentence2, Value):
                    continue
                if not pa.types.is_string(sentence1.pa_type):
                    continue
                if not pa.types.is_string(sentence2.pa_type):
                    continue
                features.pop(key1)
                features.pop(key2)
                assert "sentence" not in features, "Only one value of combining is supported"
                features["sentence"] = SentencePairFeature(key1, key2)

        # for example in dataset:
        #     yield self.text_to_instance(example, features)

        # return

        # examples = []
        # dataset.map(lambda example: examples.append(self.text_to_instance(example, features)),
        #             keep_in_memory=True,
        #             num_proc=os.cpu_count(),
        #             new_fingerprint=None)

        # for _ in range(8):
        #     print(' ' * 80 + '\n',file=sys.stderr)
        # print(len(examples))

        # return examples

        # Dataset.map()
        # examples = []

        dataset = dataset.map(
            lambda y: self._pre_tokenize(y, features=features),
            keep_in_memory=True,
            batched=True,
            num_proc=8,
            new_fingerprint=None,
            batch_size=1024,
        )

        # If we are going to reduce the number of samples, at least
        # let's do it randomly
        if self.max_instances is not None:
            dataset = dataset.shuffle(keep_in_memory=True)

        for example in dataset:
            yield self.text_to_instance(example, features)
        return
        # for example in dataset.map(self._pre_tokenize,
        #                            batched=True,
        #                            fn_kwargs=dict(features=features)):

        examples = []
        dataset.map(
            lambda example: examples.append(self.text_to_instance(example, features)),
            keep_in_memory=True,
            num_proc=1,
            new_fingerprint=None,
        )
        # for example in :
        #     yield self.text_to_instance(example, features)
        return examples

    def _batch_tokenize(
        self, batch_texts1: List[str], batch_texts2: List[str] = None
    ) -> List[List[Token]]:

        batch_tokens1 = self._tokenizer.batch_tokenize(batch_texts1)

        if batch_texts2 is not None:
            if self._ensure_whitespace_between:
                batch_texts2 = [" " + text for text in batch_texts2]
            batch_tokens2 = self._tokenizer.batch_tokenize(batch_texts2)
        else:
            batch_tokens2 = None
        # batch_tokens = [[token.text for token in tokens]
        #                 for tokens in batch_tokens]
        return self._batch_add_special_tokens(batch_tokens1, batch_tokens2)
        # return [pickle.dumps(tokens) for tokens in batch_tokens]

    def _batch_add_special_tokens(
        self, batch_tokens1: List[List[Token]], batch_tokens2: List[Optional[List[Token]]] = None
    ) -> List[List[Token]]:
        if batch_tokens2 is None:
            batch_tokens2 = [None for _ in batch_tokens1]
        return [
            self._tokenizer.add_special_tokens(tokens1, tokens2)
            for tokens1, tokens2 in zip(batch_tokens1, batch_tokens2)
        ]

    def _pre_tokenize(
        self, batch_examples: Dict[str, List[Any]], features: Features
    ) -> Dict[str, List[Any]]:
        """
        This is a helper function to perform a fast batched tokenization
        prior to more complex preprocessing and data preparation
        """
        for key, feature in features.items():
            if isinstance(feature, Value) and pa.types.is_string(feature.pa_type):
                batch_examples[f"_raw_{key}"] = batch_examples[key]
                batch_tokens = self._batch_tokenize(batch_examples[key])
            elif isinstance(feature, SentencePairFeature):
                key1, key2 = feature.key1, feature.key2
                tokens1 = batch_examples.pop(key1)
                tokens2 = batch_examples.pop(key2)
                batch_examples[f"_raw_{key1}"] = tokens1
                batch_examples[f"_raw_{key2}"] = tokens2
                batch_tokens = self._batch_tokenize(tokens1, tokens2)
            else:
                continue
            batch_examples[key] = [
                pickle.dumps(tokens, protocol=pickle.HIGHEST_PROTOCOL)
                # [dataclasses.asdict(token) for token in tokens]
                for tokens in batch_tokens
            ]

        return batch_examples

    def _string_value_to_field(
        self,
        tokens: Union[
            bytes,
            #   List[Dict[str, Any]],
            List[Token],
        ],
    ) -> TextField:
        if isinstance(tokens, bytes):
            tokens = cast(List[Token], pickle.loads(tokens))
        # if isinstance(tokens[0], dict):
        #     tokens = [Token(**kwargs) for kwargs in tokens]
        # Tokenize if not tokenized yet
        if isinstance(tokens, str):
            tokens = self._tokenizer.tokenize(tokens)
            tokens = self._tokenizer.add_special_tokens(tokens)
        return TextField(tokens, token_indexers=self._token_indexers)

    def _string_pair_to_field(
        self, tokens1: Union[bytes, List[Token]], tokens2: Union[bytes, List[Token]]
    ) -> TextField:
        if isinstance(tokens1, bytes):
            tokens1 = cast(List[Token], pickle.loads(tokens1))
        if isinstance(tokens2, bytes):
            tokens2 = cast(List[Token], pickle.loads(tokens2))

        assert isinstance(tokens1, list)
        assert isinstance(tokens2, list)
        # Tokenize if not tokenized yet
        # if isinstance(tokens1, str) and isinstance(tokens2, str):
        #     tokens1 = self._tokenizer.tokenize(tokens1)
        #     if self._ensure_whitespace_between:
        #         tokens2 = ' ' + tokens2
        #     tokens2 = self._tokenizer.tokenize(tokens2)
        # if len(tokens1) > 500:
        #     print(len(tokens1))
        #     tokens1 = tokens1[:500]
        tokens = self._tokenizer.add_special_tokens(tokens1, tokens2)
        return TextField(tokens, self._token_indexers)

    @overrides
    def text_to_instance(
        self, example: Dict[str, Any], features: Dict[str, Any], **metadata
    ) -> Instance:
        # from datasets.features import Value, ClassLabel

        # features = features.copy()

        fields: Dict[str, Field] = {}
        # return Instance(fields)

        metadata["_example"] = example
        metadata["_features"] = features

        raw: Dict[str, Any] = {}
        metadata["_raw"] = raw

        for key, value in metadata.items():
            fields[key] = MetadataField(value)

        for key, feature in features.items():
            value = example.get(key)
            field: Optional[Field]
            if 'label' in key:
                # print(value)
                assert isinstance(value, int)
                if value >= 0:
                    # label = feature.int2str(value)
                    field = LabelField(
                        label=self.class_mappings[self.scenario][value],
                        label_id=value,
                        label_namespace=f"{key}s",
                        skip_indexing=True,
                    )
                    raw[key] = value
                    # label_namespace = f'{key}s'
                    # field = LabelField(label, label_namespace)
                else:
                    field = None
            # TODO smooth labels
            elif isinstance(feature, SentencePairFeature):
                if feature.key1 in example and feature.key2 in example:
                    field = self._string_pair_to_field(example[feature.key1], example[feature.key2])
                else:
                    field = self._string_value_to_field(value)
                raw[feature.key1] = example.get(f"_raw_{feature.key1}", example.get(feature.key1))
                raw[feature.key2] = example.get(f"_raw_{feature.key2}", example.get(feature.key1))

            elif isinstance(feature, Value) and pa.types.is_string(feature.pa_type):
                field = self._string_value_to_field(value)
                raw[key] = value
            # elif isinstance(feature, Value) and pa.types.is_boolean(feature.pa_type):
            #     ...
            # elif isinstance(feature, Value) and pa.types.is_integer(feature.pa_type):
            #     ...
            # elif isinstance(feature, Value) and pa.types.is_floating(feature.pa_type):
            #     ...
            elif isinstance(feature, Value):
                if pa.types.is_floating(feature.pa_type):
                    field = ArrayField(np.array(value, dtype=feature.dtype))
                else:
                    field = MetadataField(value)
                raw[key] = value
            elif isinstance(feature, Sequence):
                field = ArrayField(np.array(value, dtype=feature.feature.dtype))
            else:
                raise NotImplementedError

            # Some fields may be absent in test time
            if field is not None:
                fields[key] = field

        instance = Instance(fields)
        return instance
        # if pickled:
        #     return pickle.dumps(instance)

        # if self._combine_input_fields:
        #     tokens = self._tokenizer.add_special_tokens(premise, hypothesis)
        #     fields["tokens"] = TextField(tokens, self._token_indexers)
        # else:
        #     premise_tokens = self._tokenizer.add_special_tokens(premise)
        #     hypothesis_tokens = self._tokenizer.add_special_tokens(hypothesis)
        #     fields["premise"] = TextField(premise_tokens, self._token_indexers)
        #     fields["hypothesis"] = TextField(hypothesis_tokens, self._token_indexers)

        #     metadata = {
        #         "premise_tokens": [x.text for x in premise_tokens],
        #         "hypothesis_tokens": [x.text for x in hypothesis_tokens],
        #     }
        #     fields["metadata"] = MetadataField(metadata)

        # if label:
        #     fields["label"] = LabelField(label)

        # for key, value in metadata.items():
        #     assert key not in fields
        #     fields[key] = MetadataField(value)

        # return Instance(fields)
