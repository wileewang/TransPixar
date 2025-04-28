from pathlib import Path
import os.path
import random
from collections import defaultdict
import math
import os

import numpy as np
import torch
from deepspeed.utils.logging import logger
from deepspeed import comm as dist
import datasets
from datasets.fingerprint import Hasher
from PIL import Image
import imageio
import multiprocess as mp

from utils.common import is_main_process, VIDEO_EXTENSIONS, round_to_nearest_multiple


DEBUG = False
IMAGE_SIZE_ROUND_TO_MULTIPLE = 32
NUM_PROC = min(8, os.cpu_count())


def shuffle_with_seed(l, seed=None):
    rng_state = random.getstate()
    random.seed(seed)
    random.shuffle(l)
    random.setstate(rng_state)


def process_caption_fn(shuffle_tags=False, caption_prefix=''):
    def fn(example):
        with open(example['caption_file']) as f:
            caption = f.read().strip()
        if shuffle_tags:
            tags = [tag.strip() for tag in caption.split(',')]
            random.shuffle(tags)
            caption = ', '.join(tags)
        caption = caption_prefix + caption

        example['caption'] = caption
        return example
    return fn


def _map_and_cache(dataset, map_fn, cache_dir, cache_file_prefix='', new_fingerprint_args=None, regenerate_cache=False, caching_batch_size=1, with_indices=False):
    # Do the fingerprinting ourselves, because otherwise map() does it by serializing the map function.
    # That goes poorly when the function is capturing huge models (slow, OOMs, etc).
    new_fingerprint_args = [] if new_fingerprint_args is None else new_fingerprint_args
    new_fingerprint_args.append(dataset._fingerprint)
    new_fingerprint = Hasher.hash(new_fingerprint_args)
    cache_file = cache_dir / f'{cache_file_prefix}{new_fingerprint}.arrow'
    cache_file = str(cache_file)
    dataset = dataset.map(
        map_fn,
        cache_file_name=cache_file,
        load_from_cache_file=(not regenerate_cache),
        writer_batch_size=100,
        new_fingerprint=new_fingerprint,
        remove_columns=dataset.column_names,
        batched=True,
        batch_size=caching_batch_size,
        with_indices=with_indices,
        num_proc=NUM_PROC,
    )
    dataset.set_format('torch')
    return dataset


# The smallest unit of a dataset. Represents a single size bucket from a single folder of images
# and captions on disk. Not batched; returns individual items.
class SizeBucketDataset:
    def __init__(self, metadata_dataset, directory_config, size_bucket, model_name):
        self.metadata_dataset = metadata_dataset
        self.directory_config = directory_config
        self.size_bucket = size_bucket
        self.model_name = model_name
        self.path = Path(self.directory_config['path'])
        self.cache_dir = self.path / 'cache' / self.model_name / f'cache_{size_bucket[0]}x{size_bucket[1]}x{size_bucket[2]}'
        os.makedirs(self.cache_dir, exist_ok=True)
        self.text_embedding_datasets = []
        self.num_repeats = self.directory_config.get('num_repeats', 1)

    def cache_latents(self, map_fn, regenerate_cache=False, caching_batch_size=1):
        print(f'caching latents: {self.size_bucket}')
        self.latent_dataset = _map_and_cache(
            self.metadata_dataset,
            map_fn,
            self.cache_dir,
            cache_file_prefix='latents_',
            regenerate_cache=regenerate_cache,
            caching_batch_size=caching_batch_size,
            with_indices=True,
        )
        # Shuffle again, since one media file can produce multiple training examples. E.g. video, or maybe
        # in the future data augmentation. Don't need to shuffle text embeddings since those are looked
        # up by index.
        if self.directory_config['shuffle']:
            self.latent_dataset = self.latent_dataset.shuffle(seed=123)
        # TODO: should we do dataset.flatten_indices() to make it contiguous on disk again?
        # self.latent_dataset = self.latent_dataset.flatten_indices(
        #     cache_file_name=str(self.cache_dir / 'latents_flattened.arrow')
        # )

    def cache_text_embeddings(self, map_fn, i, regenerate_cache=False, caching_batch_size=1):
        print(f'caching text embeddings: {self.size_bucket}')
        te_dataset = _map_and_cache(
            self.metadata_dataset,
            map_fn,
            self.cache_dir,
            cache_file_prefix=f'text_embeddings_{i}_',
            new_fingerprint_args=[i],
            regenerate_cache=regenerate_cache,
            caching_batch_size=caching_batch_size,
        )
        self.text_embedding_datasets.append(te_dataset)

    def add_text_embedding_dataset(self, te_dataset):
        self.text_embedding_datasets.append(te_dataset)

    def __getitem__(self, idx):
        idx = idx % len(self.latent_dataset)
        ret = self.latent_dataset[idx]
        te_idx = ret['te_idx'].item()
        if DEBUG:
            print(Path(self.metadata_dataset[te_idx]['image_file']).stem)
        for ds in self.text_embedding_datasets:
            ret.update(ds[te_idx])
        ret['caption'] = self.metadata_dataset[te_idx]['caption']
        return ret

    def __len__(self):
        return len(self.latent_dataset) * self.num_repeats


class PairedLatentDataset(torch.utils.data.Dataset):
    def __init__(self, latent_dataset, aux_latent_dataset):
        assert len(latent_dataset) == len(aux_latent_dataset), "Datasets must be the same length."
        self.latent_dataset = latent_dataset
        self.aux_latent_dataset = aux_latent_dataset

    def __len__(self):
        return len(self.latent_dataset)

    def __getitem__(self, idx):
        # Get the primary latent data and its auxiliary latent
        primary = self.latent_dataset[idx]
        aux = self.aux_latent_dataset[idx]
        
        # Merge the two dictionaries (or create a new one)
        merged = primary.copy()
        merged['aux_latents'] = aux['latents']
        return merged


class PairSizeBucketDataset(SizeBucketDataset):
    def __init__(self, metadata_dataset, directory_config, size_bucket, model_name):

        assert isinstance(metadata_dataset, list)

        self.metadata_dataset = metadata_dataset[0]
        self.metadata_dataset_aux = metadata_dataset[1]

        self.directory_config = directory_config
        self.size_bucket = size_bucket
        self.model_name = model_name
        self.path = Path(self.directory_config['path'])
        self.cache_dir = self.path / 'cache' / self.model_name / f'cache_{size_bucket[0]}x{size_bucket[1]}x{size_bucket[2]}'
        os.makedirs(self.cache_dir, exist_ok=True)
        self.text_embedding_datasets = []
        self.num_repeats = self.directory_config.get('num_repeats', 1)
    
    def cache_latents(self, map_fn, regenerate_cache=False, caching_batch_size=1):
        print(f'caching latents: {self.size_bucket}')
        self.latent_dataset = _map_and_cache(
            self.metadata_dataset,
            map_fn,
            self.cache_dir,
            cache_file_prefix='latents_',
            regenerate_cache=regenerate_cache,
            caching_batch_size=caching_batch_size,
            with_indices=True,
        )

        self.aux_latent_dataset = _map_and_cache(
            self.metadata_dataset_aux,
            map_fn,
            self.cache_dir,
            cache_file_prefix='aux_latents_',
            regenerate_cache=regenerate_cache,
            caching_batch_size=caching_batch_size,
            with_indices=True,
        )
        # Shuffle again, since one media file can produce multiple training examples. E.g. video, or maybe
        # in the future data augmentation. Don't need to shuffle text embeddings since those are looked
        # up by index.
        if self.directory_config['shuffle']:
            self.latent_dataset = self.latent_dataset.shuffle(seed=123)
            self.aux_latent_dataset = self.aux_latent_dataset.shuffle(seed=123)
        
        # Merge the paired data into one dataset
        self.paired_dataset = PairedLatentDataset(self.latent_dataset, self.aux_latent_dataset)

        # TODO: should we do dataset.flatten_indices() to make it contiguous on disk again?
        # self.latent_dataset = self.latent_dataset.flatten_indices(
        #     cache_file_name=str(self.cache_dir / 'latents_flattened.arrow')
        # )


    # def __getitem__(self, idx):
    #     idx = idx % len(self.latent_dataset)
    #     ret = self.latent_dataset[idx]
    #     aux_data = self.aux_latent_dataset[idx]
    #     ret['aux_latents'] = aux_data['latents']

    #     te_idx = ret['te_idx'].item()
    #     if DEBUG:
    #         print(Path(self.metadata_dataset[te_idx]['image_file']).stem)
    #     for ds in self.text_embedding_datasets:
    #         ret.update(ds[te_idx])
    #     ret['caption'] = self.metadata_dataset[te_idx]['caption']
    #     return ret

    def __getitem__(self, idx):
        idx = idx % len(self.paired_dataset)
        ret = self.paired_dataset[idx]
        
        # If you have additional text embedding datasets that use the index from the metadata,
        # make sure to update the result accordingly.
        te_idx = ret['te_idx'].item()  # if needed for text embeddings
        for ds in self.text_embedding_datasets:
            ret.update(ds[te_idx])
        ret['caption'] = self.metadata_dataset[te_idx]['caption']
        return ret



# Logical concatenation of multiple SizeBucketDataset, for the same size bucket. It returns items
# as batches.
class ConcatenatedBatchedDataset:
    def __init__(self, datasets):
        self.datasets = datasets
        self.post_init_called = False

    def post_init(self, batch_size):
        iteration_order = []
        for i, ds in enumerate(self.datasets):
            iteration_order.extend([i]*len(ds))
        shuffle_with_seed(iteration_order, 0)
        cumulative_sums = [0] * len(self.datasets)
        for k, dataset_idx in enumerate(iteration_order):
            iteration_order[k] = (dataset_idx, cumulative_sums[dataset_idx])
            cumulative_sums[dataset_idx] += 1
        self.iteration_order = iteration_order
        self.batch_size = batch_size
        self._make_divisible_by(self.batch_size)
        self.post_init_called = True

    def __len__(self):
        assert self.post_init_called
        return len(self.iteration_order) // self.batch_size

    def __getitem__(self, idx):
        assert self.post_init_called
        start = idx * self.batch_size
        end = start + self.batch_size
        return [self.datasets[i][j] for i, j in self.iteration_order[start:end]]

    def _make_divisible_by(self, n):
        new_length = (len(self.iteration_order) // n) * n
        self.iteration_order = self.iteration_order[:new_length]
        if new_length == 0 and is_main_process():
            logger.warning(f"size bucket {self.datasets[0].size_bucket} is being completely dropped because it doesn't have enough images")


class ARBucketDataset:
    def __init__(self, ar_frames, resolutions, metadata_dataset, directory_config, model_name):
        self.ar_frames = ar_frames
        self.resolutions = resolutions
        self.metadata_dataset = metadata_dataset
        self.directory_config = directory_config
        self.model_name = model_name
        self.size_buckets = []
        self.path = Path(directory_config['path'])
        self.cache_dir = self.path / 'cache' / self.model_name / f'ar_frames_{self.ar_frames[0]:.3f}_{self.ar_frames[1]}'
        os.makedirs(self.cache_dir, exist_ok=True)

        for res in resolutions:
            area = res**2
            w = math.sqrt(area * self.ar_frames[0])
            h = area / w
            w = round_to_nearest_multiple(w, IMAGE_SIZE_ROUND_TO_MULTIPLE)
            h = round_to_nearest_multiple(h, IMAGE_SIZE_ROUND_TO_MULTIPLE)
            size_bucket = (w, h, self.ar_frames[1])
            metadata_with_size_bucket = self.metadata_dataset.map(lambda example: {'size_bucket': size_bucket}, keep_in_memory=True)
            self.size_buckets.append(
                SizeBucketDataset(metadata_with_size_bucket, directory_config, size_bucket, model_name)
            )

    def get_size_bucket_datasets(self):
        return self.size_buckets

    def cache_latents(self, map_fn, regenerate_cache=False, caching_batch_size=1):
        print(f'caching latents: {self.ar_frames}')
        for ds in self.size_buckets:
            ds.cache_latents(map_fn, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)

    def cache_text_embeddings(self, map_fn, i, regenerate_cache=False, caching_batch_size=1):
        print(f'caching text embeddings: {self.ar_frames}')
        te_dataset = _map_and_cache(
            self.metadata_dataset,
            map_fn,
            self.cache_dir,
            cache_file_prefix=f'text_embeddings_{i}_',
            new_fingerprint_args=[i],
            regenerate_cache=regenerate_cache,
            caching_batch_size=caching_batch_size,
        )
        for size_bucket_dataset in self.size_buckets:
            size_bucket_dataset.add_text_embedding_dataset(te_dataset)


class PairARBucketDataset(ARBucketDataset):
    def __init__(self, ar_frames, resolutions, metadata_dataset, directory_config, model_name):
        self.ar_frames = ar_frames
        self.resolutions = resolutions

        self.metadata_dataset = metadata_dataset[0]
        self.metadata_dataset_aux = metadata_dataset[1]

        self.directory_config = directory_config
        self.model_name = model_name
        self.size_buckets = []
        self.path = Path(directory_config['path'])
        self.cache_dir = self.path / 'cache' / self.model_name / f'ar_frames_{self.ar_frames[0]:.3f}_{self.ar_frames[1]}'
        os.makedirs(self.cache_dir, exist_ok=True)

        for res in resolutions:
            area = res**2
            w = math.sqrt(area * self.ar_frames[0])
            h = area / w
            w = round_to_nearest_multiple(w, IMAGE_SIZE_ROUND_TO_MULTIPLE)
            h = round_to_nearest_multiple(h, IMAGE_SIZE_ROUND_TO_MULTIPLE)
            size_bucket = (w, h, self.ar_frames[1])

            metadata_with_size_bucket = self.metadata_dataset.map(lambda example: {'size_bucket': size_bucket}, keep_in_memory=True)
            metadata_with_size_bucket_aux = self.metadata_dataset_aux.map(lambda example: {'size_bucket': size_bucket}, keep_in_memory=True)

            self.size_buckets.append(
                PairSizeBucketDataset([metadata_with_size_bucket, metadata_with_size_bucket_aux], directory_config, size_bucket, model_name)
            )
    

class DirectoryDataset:
    def __init__(self, directory_config, dataset_config, model_name, framerate=None, skip_dataset_validation=False):
        self._set_defaults(directory_config, dataset_config)
        self.directory_config = directory_config
        self.dataset_config = dataset_config
        if not skip_dataset_validation:
            self.validate()
        self.model_name = model_name
        self.framerate = framerate
        self.enable_ar_bucket = directory_config.get('enable_ar_bucket', dataset_config.get('enable_ar_bucket', False))
        # Configure directly from user-specified size buckets.
        self.size_buckets = directory_config.get('size_buckets', dataset_config.get('size_buckets', None))
        self.use_size_buckets = (self.size_buckets is not None)
        if self.use_size_buckets:
            # sort size bucket from longest frame length to shortest
            self.size_buckets.sort(key=lambda t: t[-1], reverse=True)
            self.size_buckets = np.array(self.size_buckets)
        else:
            self.resolutions = self._process_user_provided_resolutions(
                directory_config.get('resolutions', dataset_config['resolutions'])
            )
        self.path = Path(self.directory_config['path'])
        self.cache_dir = self.path / 'cache' / self.model_name

        if not self.path.exists() or not self.path.is_dir():
            raise RuntimeError(f'Invalid path: {self.path}')

        if self.use_size_buckets:
            self.ars = np.array([w / h for w, h, _ in self.size_buckets])
        elif not self.enable_ar_bucket:
            self.ars = np.array([1.0])
        elif ars := self.directory_config.get('ar_buckets', self.dataset_config.get('ar_buckets', None)):
            self.ars = self._process_user_provided_ars(ars)
        else:
            min_ar = self.directory_config.get('min_ar', self.dataset_config['min_ar'])
            max_ar = self.directory_config.get('max_ar', self.dataset_config['max_ar'])
            num_ar_buckets = self.directory_config.get('num_ar_buckets', self.dataset_config['num_ar_buckets'])
            self.ars = np.geomspace(min_ar, max_ar, num=num_ar_buckets)
        self.log_ars = np.log(self.ars)
        frame_buckets = self.directory_config.get('frame_buckets', self.dataset_config.get('frame_buckets', [1]))
        if 1 not in frame_buckets:
            # always have an image bucket for convenience
            frame_buckets.append(1)
        frame_buckets.sort()
        self.frame_buckets = np.array(frame_buckets)

    def validate(self):
        resolutions = self.directory_config.get('resolutions', self.dataset_config.get('resolutions', []))
        if len(resolutions) > 3:
            if is_main_process():
                print('WARNING: You have set a lot of resolutions in the dataset config. Please read the comments in the example dataset.toml file,'
                      ' and make sure you understand what this setting does. If you still want to proceed with the current configuration,'
                      ' run the script with the --i_know_what_i_am_doing flag.')
            quit()

    def cache_metadata(self, regenerate_cache=False):
        files = list(self.path.glob('*'))
        # deterministic order
        files.sort()

        image_files = []
        caption_files = []
        for file in files:
            if not file.is_file() or file.suffix == '.txt' or file.suffix == '.npz':
                continue
            image_file = file
            caption_file = image_file.with_suffix('.txt')
            if not os.path.exists(caption_file):
                logger.warning(f'Image file {image_file} does not have corresponding caption file.')
                caption_file = ''
            image_files.append(str(image_file))
            caption_files.append(str(caption_file))
        assert len(image_files) > 0, f'Directory {self.path} had no images/videos!'

        metadata_dataset = datasets.Dataset.from_dict({'image_file': image_files, 'caption_file': caption_files})
        # Shuffle the data. Use a fixed seed, so the dataset is identical on all processes.
        # Processes other than rank 0 will then load it from cache.
        metadata_dataset = metadata_dataset.shuffle(seed=0)
        metadata_map_fn = self._metadata_map_fn()
        fingerprint = Hasher.hash([metadata_dataset._fingerprint, metadata_map_fn])
        print('caching metadata')
        metadata_dataset = metadata_dataset.map(
            metadata_map_fn,
            cache_file_name=str(self.cache_dir / f'metadata/metadata_{fingerprint}.arrow'),
            load_from_cache_file=(not regenerate_cache),
            batched=True,
            batch_size=1,
            num_proc=NUM_PROC,
            remove_columns=metadata_dataset.column_names,
        )

        grouped_metadata = defaultdict(lambda: defaultdict(list))
        for example in metadata_dataset:
            if self.use_size_buckets:
                grouping_key = tuple(example['size_bucket'])
            else:
                grouping_key = example['ar_bucket']
                grouping_key = (grouping_key[0], int(grouping_key[1]))
            d = grouped_metadata[grouping_key]
            for k, v in example.items():
                d[k].append(v)

        if self.use_size_buckets:
            self.size_bucket_datasets = []
            for size_bucket, metadata in grouped_metadata.items():
                metadata = datasets.Dataset.from_dict(metadata)
                self.size_bucket_datasets.append(
                    SizeBucketDataset(
                        metadata,
                        self.directory_config,
                        size_bucket,
                        self.model_name,
                    )
                )
        else:
            self.ar_bucket_datasets = []
            for ar_bucket, metadata in grouped_metadata.items():
                metadata = datasets.Dataset.from_dict(metadata)
                self.ar_bucket_datasets.append(
                    ARBucketDataset(
                        ar_bucket,
                        self.resolutions,
                        metadata,
                        self.directory_config,
                        self.model_name,
                    )
                )

    def _set_defaults(self, directory_config, dataset_config):
        directory_config.setdefault('enable_ar_bucket', dataset_config.get('enable_ar_bucket', False))
        directory_config.setdefault('shuffle_tags', dataset_config.get('shuffle_tags', False))
        directory_config.setdefault('caption_prefix', dataset_config.get('caption_prefix', ''))

    def _metadata_map_fn(self):
        def fn(example):
            # batch size always 1
            caption_file = example['caption_file'][0]
            image_file = example['image_file'][0]
            if not caption_file:
                caption = ''
            else:
                with open(caption_file) as f:
                    caption = f.read().strip()
            if self.directory_config['shuffle_tags']:
                tags = [tag.strip() for tag in caption.split(',')]
                random.shuffle(tags)
                caption = ', '.join(tags)
            caption = self.directory_config['caption_prefix'] + caption
            empty_return = {'image_file': [], 'caption': [], 'ar_bucket': [], 'size_bucket': [], 'is_video': []}

            image_file = Path(image_file)
            if image_file.suffix == '.webp':
                frames = imageio.get_reader(image_file).get_length()
                if frames > 1:
                    raise NotImplementedError('WebP videos are not supported.')
            try:
                if image_file.suffix in VIDEO_EXTENSIONS:
                    # 100% accurate frame count, but much slower.
                    # frames = 0
                    # for frame in imageio.v3.imiter(image_file):
                    #     frames += 1
                    #     height, width = frame.shape[:2]
                    # TODO: this is an estimate of frame count. What happens if variable frame rate? Is
                    # it still close enough?
                    meta = imageio.v3.immeta(image_file)
                    first_frame = next(imageio.v3.imiter(image_file))
                    height, width = first_frame.shape[:2]
                    assert self.framerate is not None, "Need model framerate but don't have it. This shouldn't happen. Is the framerate attribute on the model set?"
                    frames = round(self.framerate * meta['duration']) # Luozhou 根据长度来计算frames, 修正为向上取整
                else:
                    pil_img = Image.open(image_file)
                    width, height = pil_img.size
                    frames = 1
            except Exception:
                logger.warning(f'Media file {image_file} could not be opened. Skipping.')
                return empty_return
            is_video = (frames > 1)
            log_ar = np.log(width / height)

            if self.use_size_buckets:
                size_bucket = self._find_closest_size_bucket(log_ar, frames, is_video)
                if size_bucket is None:
                    print(f'video with frames={frames} is being skipped because it is too short')
                    return empty_return
                ar_bucket = None
            else:
                ar_bucket = self._find_closest_ar_bucket(log_ar, frames, is_video)
                if ar_bucket is None:
                    print(f'video with frames={frames} is being skipped because it is too short')
                    return empty_return
                size_bucket = None

            return {'image_file': [str(image_file)], 'caption': [caption], 'ar_bucket': [ar_bucket], 'size_bucket': [size_bucket], 'is_video': [is_video]}
        return fn

    def _find_closest_ar_bucket(self, log_ar, frames, is_video):
        # Best AR bucket is the one with the smallest AR difference in log space.
        i = np.argmin(np.abs(log_ar - self.log_ars))
        # find closest frame bucket where the number of frames is greater than or equal to the bucket
        diffs = frames - self.frame_buckets
        positive_diffs = diffs[diffs >= 0]
        if len(positive_diffs) == 0:
            # video not long enough to find any valid frame bucket
            return None
        j = np.argmin(positive_diffs)
        if is_video and self.frame_buckets[j] == 1:
            # don't let video be mapped to the image frame bucket
            return None
        ar_bucket = (self.ars[i], self.frame_buckets[j])
        return ar_bucket

    def _find_closest_size_bucket(self, log_ar, frames, is_video):
        # Best AR bucket is the one with the smallest AR difference in log space.
        ar_diffs = np.abs(log_ar - self.log_ars)
        candidate_size_buckets = self.size_buckets[np.argsort(ar_diffs, kind='stable')]
        # Find closest size bucket where the number of frames is greater than or equal to the bucket.
        # self.size_buckets was already sorted longest -> shortest frame length
        found = False
        for size_bucket in candidate_size_buckets:
            if is_video and size_bucket[-1] == 1:
                # don't let video be mapped to the image frame bucket
                continue
            if frames >= size_bucket[-1]:
                found = True
                break
        if not found:
            # video not long enough to find any valid frame bucket
            return None
        return size_bucket

    def _process_user_provided_ars(self, ars):
        ar_buckets = set()
        for ar in ars:
            if isinstance(ar, (tuple, list)):
                assert len(ar) == 2
                ar = round(ar[0] / ar[1], 6)
            ar_buckets.add(ar)
        ar_buckets = list(ar_buckets)
        ar_buckets.sort()
        return np.array(ar_buckets)

    def _process_user_provided_resolutions(self, resolutions):
        result = set()
        for res in resolutions:
            if isinstance(res, (tuple, list)):
                assert len(res) == 2
                res = round(math.sqrt(res[0] * res[1]), 6)
            result.add(res)
        result = list(result)
        result.sort()
        return result

    def get_size_bucket_datasets(self):
        if self.use_size_buckets:
            return self.size_bucket_datasets
        result = []
        for ar_bucket_dataset in self.ar_bucket_datasets:
            result.extend(ar_bucket_dataset.get_size_bucket_datasets())
        return result

    def cache_latents(self, map_fn, regenerate_cache=False, caching_batch_size=1):
        print(f'caching latents: {self.path}')
        datasets = self.size_bucket_datasets if self.use_size_buckets else self.ar_bucket_datasets
        for ds in datasets:
            ds.cache_latents(map_fn, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)

    def cache_text_embeddings(self, map_fn, i, regenerate_cache=False, caching_batch_size=1):
        print(f'caching text embeddings: {self.path}')
        datasets = self.size_bucket_datasets if self.use_size_buckets else self.ar_bucket_datasets
        for ds in datasets:
            ds.cache_text_embeddings(map_fn, i, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)


class PairDirectoryDataset(DirectoryDataset):

    def __init__(self, directory_config, dataset_config, model_name, framerate=None, skip_dataset_validation=False):
        super().__init__(directory_config, dataset_config, model_name, framerate, skip_dataset_validation)
    
    def cache_metadata(self, regenerate_cache=False):
        files = list(self.path.glob('*'))
        files.sort()  # deterministic order

        # 用于存储：ID -> path
        image_dict = {}
        aux_image_dict = {}
        caption_dict = {}

        for file in files:
            if not file.is_file() or file.suffix in ['.txt', '.npz']:
                continue

            basename = os.path.splitext(file.name)[0]
            if basename.endswith('_seg'):
                key = basename.replace('_seg', '')
                aux_image_dict[key] = str(file)
            else:
                key = basename
                image_dict[key] = str(file)
                caption_path = file.with_suffix('.txt')
                if caption_path.exists():
                    caption_dict[key] = str(caption_path)
                else:
                    logger.warning(f'Image file {file} does not have corresponding caption file.')
                    caption_dict[key] = ''

        # 只保留两边都有的 key，并排序保证顺序一致
        common_keys = sorted(set(image_dict.keys()) & set(aux_image_dict.keys()))

        image_files = [image_dict[k] for k in common_keys]
        aux_image_files = [aux_image_dict[k] for k in common_keys]
        caption_files = [caption_dict[k] for k in common_keys]

        assert len(image_files) > 0, f'Directory {self.path} had no images/videos!'
        assert len(aux_image_files) > 0, f'Directory {self.path} had no aux videos!'

        # 可选校验：确保顺序对得上
        for i, (img, aux) in enumerate(zip(image_files, aux_image_files)):
            img_id = os.path.splitext(os.path.basename(img))[0]
            aux_id = os.path.splitext(os.path.basename(aux))[0].replace("_seg", "")
            assert img_id == aux_id, f"Mismatch at index {i}: {img_id} vs {aux_id}"



        # import json
        # import uuid
        # from datetime import datetime

        # # 生成唯一ID（可以基于时间 + uuid 确保唯一性）
        # unique_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8]

        # # 构建 metadata 列表
        # metadata = [
        #     {
        #         "image_file": img,
        #         "caption_file": cap
        #     }
        #     for img, cap in zip(image_files, caption_files)
        # ]

        # # 构建 metadata_aux 列表
        # metadata_aux = [
        #     {
        #         "image_file": img,
        #         "caption_file": cap
        #     }
        #     for img, cap in zip(aux_image_files, caption_files)
        # ]

        # # 保存路径
        # os.makedirs("/data/user/user108/projects_lzw/test-json", exist_ok=True)

        # with open(f"/data/user/user108/projects_lzw/test-json/{unique_id}_metadata.json", "w", encoding="utf-8") as f:
        #     json.dump(metadata, f, ensure_ascii=False, indent=2)

        # with open(f"/data/user/user108/projects_lzw/test-json/{unique_id}_metadata_aux.json", "w", encoding="utf-8") as f:
        #     json.dump(metadata_aux, f, ensure_ascii=False, indent=2)

        # print(f"Saved metadata files with ID: {unique_id}")


        metadata_dataset = datasets.Dataset.from_dict({'image_file': image_files, 'caption_file': caption_files})
        metadata_dataset_aux = datasets.Dataset.from_dict({'image_file': aux_image_files, 'caption_file': caption_files})

        # Shuffle the data. Use a fixed seed, so the dataset is identical on all processes.
        # Processes other than rank 0 will then load it from cache.

        if self.directory_config['shuffle']:
            metadata_dataset = metadata_dataset.shuffle(seed=0)
            metadata_dataset_aux = metadata_dataset_aux.shuffle(seed=0)

        metadata_map_fn = self._metadata_map_fn()

        fingerprint = Hasher.hash([metadata_dataset._fingerprint, metadata_map_fn])
        # fingerprint_aux = Hasher.hash([metadata_dataset_aux._fingerprint, metadata_map_fn])

        print('caching metadata')
        metadata_dataset = metadata_dataset.map(
            metadata_map_fn,
            cache_file_name=str(self.cache_dir / f'metadata/metadata_{fingerprint}.arrow'),
            load_from_cache_file=(not regenerate_cache),
            batched=True,
            batch_size=1,
            num_proc=NUM_PROC,
            remove_columns=metadata_dataset.column_names,
        )

        metadata_dataset_aux = metadata_dataset_aux.map(
            metadata_map_fn,
            cache_file_name=str(self.cache_dir / f'metadata/metadata_{fingerprint}_aux.arrow'),
            load_from_cache_file=(not regenerate_cache),
            batched=True,
            batch_size=1,
            num_proc=NUM_PROC,
            remove_columns=metadata_dataset_aux.column_names,
        )

        grouped_metadata = defaultdict(lambda: defaultdict(list))
        for example in metadata_dataset:
            if self.use_size_buckets:
                grouping_key = tuple(example['size_bucket'])
            else:
                grouping_key = example['ar_bucket']
                grouping_key = (grouping_key[0], int(grouping_key[1]))
            d = grouped_metadata[grouping_key]
            for k, v in example.items():
                d[k].append(v)
        
        # prepare for aux
        grouped_metadata_aux = defaultdict(lambda: defaultdict(list))
        for example in metadata_dataset_aux:
            if self.use_size_buckets:
                grouping_key = tuple(example['size_bucket'])
            else:
                grouping_key = example['ar_bucket']
                grouping_key = (grouping_key[0], int(grouping_key[1]))
            d = grouped_metadata_aux[grouping_key]
            for k, v in example.items():
                d[k].append(v)
        
        if self.use_size_buckets:
            self.size_bucket_datasets = []
            # for size_bucket, metadata in grouped_metadata.items():
            for (size_bucket, metadata), (_, metadata_aux) in zip(grouped_metadata.items(), grouped_metadata_aux.items()):
                metadata = datasets.Dataset.from_dict(metadata)
                metadata_aux = datasets.Dataset.from_dict(metadata_aux)
                self.size_bucket_datasets.append(
                    PairSizeBucketDataset(
                        [metadata, metadata_aux],
                        self.directory_config,
                        size_bucket,
                        self.model_name,
                    )
                )
        else:
            # raise NotImplementedError
            self.ar_bucket_datasets = []
            # for ar_bucket, metadata in grouped_metadata.items():
            for (ar_bucket, metadata), (_, metadata_aux) in zip(grouped_metadata.items(), grouped_metadata_aux.items()):
                # import json
                # import uuid

                # def safe_filename(s):
                #     return "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in s)

                # # 根据 ar_bucket 生成一个基础字符串（ar_bucket 是 tuple，可以用 join 转换成字符串）
                # ar_bucket_str = safe_filename("_".join(str(x) for x in ar_bucket))
                # # 使用 ar_bucket_str 加上随机 uuid 生成唯一文件标识
                # unique_id = f"{ar_bucket_str}_{uuid.uuid4().hex[:8]}"
                
                # # 定义两个 JSON 文件的完整路径
                # metadata_file = os.path.join("/hpc2hdd/home/lwang592/projects/video-generation-runs/test-json", f"{unique_id}_metadata.json")
                # metadata_aux_file = os.path.join("/hpc2hdd/home/lwang592/projects/video-generation-runs/test-json", f"{unique_id}_metadata_aux.json")
                
                # # 保存 metadata 数据
                # with open(metadata_file, "w", encoding="utf-8") as f:
                #     json.dump(metadata, f, ensure_ascii=False, indent=2)
                # # 保存 metadata_aux 数据
                # with open(metadata_aux_file, "w", encoding="utf-8") as f:
                #     json.dump(metadata_aux, f, ensure_ascii=False, indent=2)

                metadata = datasets.Dataset.from_dict(metadata)
                metadata_aux = datasets.Dataset.from_dict(metadata_aux)
                self.ar_bucket_datasets.append(
                    PairARBucketDataset(
                        ar_bucket,
                        self.resolutions,
                        [metadata, metadata_aux],
                        self.directory_config,
                        self.model_name,
                    )
                )

# Outermost dataset object that the caller uses. Contains multiple ConcatenatedBatchedDataset. Responsible
# for returning the correct batch for the process's data parallel rank. Calls model.prepare_inputs so the
# returned tuple of tensors is whatever the model needs.
class Dataset:
    def __init__(self, dataset_config, model, skip_dataset_validation=False):
        super().__init__()
        self.dataset_config = dataset_config
        self.model = model
        self.model_name = self.model.name
        self.post_init_called = False
        self.eval_quantile = None
        if not skip_dataset_validation:
            self.model.model_specific_dataset_config_validation(self.dataset_config)

        self.directory_datasets = []
        for directory_config in dataset_config['directory']:
            if directory_config.get("mode", "base") == "pair":
                directory_dataset = PairDirectoryDataset(
                    directory_config,
                    dataset_config,
                    self.model_name,
                    framerate=model.framerate,
                    skip_dataset_validation=skip_dataset_validation,
                )
            else:
                directory_dataset = DirectoryDataset(
                    directory_config,
                    dataset_config,
                    self.model_name,
                    framerate=model.framerate,
                    skip_dataset_validation=skip_dataset_validation,
                )
            self.directory_datasets.append(directory_dataset)

    def post_init(self, data_parallel_rank, data_parallel_world_size, per_device_batch_size, gradient_accumulation_steps):
        self.data_parallel_rank = data_parallel_rank
        self.data_parallel_world_size = data_parallel_world_size
        self.batch_size = per_device_batch_size * gradient_accumulation_steps
        self.global_batch_size = self.data_parallel_world_size * self.batch_size

        # group same size_bucket together
        datasets_by_size_bucket = defaultdict(list)
        for directory_dataset in self.directory_datasets:
            for size_bucket_dataset in directory_dataset.get_size_bucket_datasets():
                datasets_by_size_bucket[size_bucket_dataset.size_bucket].append(size_bucket_dataset)
        self.buckets = []
        for datasets in datasets_by_size_bucket.values():
            self.buckets.append(ConcatenatedBatchedDataset(datasets))

        for bucket in self.buckets:
            bucket.post_init(self.global_batch_size)

        iteration_order = []
        for i, bucket in enumerate(self.buckets):
            iteration_order.extend([i]*(len(bucket)))
        shuffle_with_seed(iteration_order, 0)
        cumulative_sums = [0] * len(self.buckets)
        for k, dataset_idx in enumerate(iteration_order):
            iteration_order[k] = (dataset_idx, cumulative_sums[dataset_idx])
            cumulative_sums[dataset_idx] += 1
        self.iteration_order = iteration_order
        if DEBUG:
            print(f'Dataset iteration_order: {self.iteration_order}')

        self.post_init_called = True

        if subsample_ratio := self.dataset_config.get('subsample_ratio', None):
            new_len = int(len(self) * subsample_ratio)
            self.iteration_order = self.iteration_order[:new_len]

    def set_eval_quantile(self, quantile):
        self.eval_quantile = quantile

    def __len__(self):
        assert self.post_init_called
        return len(self.iteration_order)

    def __getitem__(self, idx):
        assert self.post_init_called
        i, j = self.iteration_order[idx]
        examples = self.buckets[i][j]
        start_idx = self.data_parallel_rank*self.batch_size
        examples_for_this_dp_rank = examples[start_idx:start_idx+self.batch_size]
        if DEBUG:
            print((start_idx, start_idx+self.batch_size))
        batch = self._collate(examples_for_this_dp_rank)
        return batch

    # Collates a list of feature dictionaries into a single dictionary of batched features.
    # Each feature can be a tensor, list, or single item.
    def _collate(self, examples):
        ret = {}
        for key, value in examples[0].items():
            if torch.is_tensor(value):
                ret[key] = torch.stack([example[key] for example in examples])
            else:
                ret[key] = [example[key] for example in examples]
        return ret

    def cache_metadata(self, regenerate_cache=False):
        for ds in self.directory_datasets:
            ds.cache_metadata(regenerate_cache=regenerate_cache)

    def cache_latents(self, map_fn, regenerate_cache=False, caching_batch_size=1):
        for ds in self.directory_datasets:
            ds.cache_latents(map_fn, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)

    def cache_text_embeddings(self, map_fn, i, regenerate_cache=False, caching_batch_size=1):
        for ds in self.directory_datasets:
            ds.cache_text_embeddings(map_fn, i, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)


def _cache_fn(datasets, queue, preprocess_media_file_fn, num_text_encoders, regenerate_cache, caching_batch_size):
    # Dataset map() starts a bunch of processes. Make sure torch uses a limited number of threads
    # to avoid CPU contention.
    # TODO: if we ever change Datasets map to use spawn instead of fork, this might not work.
    #torch.set_num_threads(os.cpu_count() // NUM_PROC)
    # HF Datasets map can randomly hang if this is greater than one (???)
    # See https://github.com/pytorch/pytorch/issues/10996
    # Alternatively, we could try fixing this by using spawn instead of fork.
    torch.set_num_threads(1)

    for ds in datasets:
        ds.cache_metadata(regenerate_cache=regenerate_cache)

    def latents_map_fn(example, indices): # Luozhou default map function
        first_size_bucket = example['size_bucket'][0]
        tensors = []
        te_idx = []
        for idx, path, size_bucket in zip(indices, example['image_file'], example['size_bucket']):
            assert size_bucket == first_size_bucket
            items = preprocess_media_file_fn(path, size_bucket)
            tensors.extend(items)
            te_idx.extend([idx] * len(items))

        if len(tensors) == 0:
            return {'latents': [], 'te_idx': []}

        caching_batch_size = len(example['image_file'])
        results = defaultdict(list)
        for i in range(0, len(tensors), caching_batch_size):
            batched = torch.stack(tensors[i:i+caching_batch_size])
            parent_conn, child_conn = mp.Pipe(duplex=False)
            queue.put((0, batched, child_conn))
            result = parent_conn.recv()  # dict
            for k, v in result.items():
                results[k].append(v)
        # concatenate the list of tensors at each key into one batched tensor
        for k, v in results.items():
            results[k] = torch.cat(v)
        results['te_idx'] = te_idx
        return results

    for ds in datasets:
        ds.cache_latents(latents_map_fn, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)

    for text_encoder_idx in range(num_text_encoders):
        def text_embedding_map_fn(example):
            parent_conn, child_conn = mp.Pipe(duplex=False)
            queue.put((text_encoder_idx+1, example['caption'], example['is_video'], child_conn))
            result = parent_conn.recv()  # dict
            return result
        for ds in datasets:
            ds.cache_text_embeddings(text_embedding_map_fn, text_encoder_idx+1, regenerate_cache=regenerate_cache, caching_batch_size=caching_batch_size)

    # signal that we're done
    queue.put(None)


# Helper class to make caching multiple datasets more efficient by moving
# models to GPU as few times as needed.
class DatasetManager:
    def __init__(self, model, regenerate_cache=False, caching_batch_size=1):
        self.model = model
        self.vae = self.model.get_vae()
        self.text_encoders = self.model.get_text_encoders()
        self.submodels = [self.vae] + list(self.text_encoders)
        self.call_vae_fn = self.model.get_call_vae_fn(self.vae)
        self.call_text_encoder_fns = [self.model.get_call_text_encoder_fn(text_encoder) for text_encoder in self.text_encoders]
        self.regenerate_cache = regenerate_cache
        self.caching_batch_size = caching_batch_size
        self.datasets = []

    def register(self, dataset):
        self.datasets.append(dataset)

    # Some notes for myself:
    # Use a manager queue, since that can be pickled and unpickled, and sent to other processes.
    # IMPORTANT: we use multiprocess library (not Python multiprocessing!) just like HF Datasets does.
    # After hours of debugging and looking up related issues, I have concluded multiprocessing is outright bugged
    # for this use case. Something about making a manager queue and sending it to the caching process, and then
    # further sending it to map() workers via the pickled map function, is broken. It gets through a lot of the caching,
    # but eventually, inevitably, queue.put() will fail with BrokenPipeError. Switching from multiprocessing to multiprocess,
    # which has basically the same API, and everything works perfectly. ¯\_(ツ)_/¯
    def cache(self):
        if is_main_process():
            manager = mp.Manager()
            queue = [manager.Queue()]
        else:
            queue = [None]
        torch.distributed.broadcast_object_list(queue, src=0, group=dist.get_world_group())
        queue = queue[0]

        # start up a process to run through the dataset caching flow
        if is_main_process():
            process = mp.Process(
                target=_cache_fn,
                args=(
                    self.datasets,
                    queue,
                    self.model.get_preprocess_media_file_fn(),
                    len(self.text_encoders),
                    self.regenerate_cache,
                    self.caching_batch_size,
                )
            )
            process.start()

        # loop on the original processes (one per GPU) to handle tasks requiring GPU models (VAE, text encoders)
        while True:
            task = queue.get()
            if task is None:
                # Propagate None so all worker processes break out of this loop.
                # This is safe because it's a FIFO queue. The first None always comes after all work items.
                queue.put(None)
                break
            self._handle_task(task)

        # Free memory in all unneeded submodels. This is easier than trying to delete every reference.
        # TODO: check if this is actually freeing memory.
        for model in self.submodels:
            model.to('meta')

        dist.barrier()
        if is_main_process():
            process.join()

        # Now load all datasets from cache.
        for ds in self.datasets:
            ds.cache_metadata()
            ds.cache_latents(None)
            for i in range(1, len(self.text_encoders)+1):
                ds.cache_text_embeddings(None, i)

    @torch.no_grad()
    def _handle_task(self, task):
        id = task[0]
        # moved needed submodel to cuda, and everything else to cpu
        if next(self.submodels[id].parameters()).device.type != 'cuda':
            for i, submodel in enumerate(self.submodels):
                if i != id:
                    submodel.to('cpu')
            self.submodels[id].to('cuda')
        if id == 0:
            tensor, pipe = task[1:]
            results = self.call_vae_fn(tensor)
        elif id > 0:
            caption, is_video, pipe = task[1:]
            results = self.call_text_encoder_fns[id-1](caption, is_video=is_video)
        else:
            raise RuntimeError()
        # Need to move to CPU here. If we don't, we get this error:
        # RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
        # I think this is because HF Datasets uses the multiprocess library (different from Python multiprocessing!) so it will always use fork.
        results = {k: v.to('cpu') for k, v in results.items()}
        pipe.send(results)


def split_batch(batch, pieces):
    example_tuple = batch
    split_size = example_tuple[0].size(0) // pieces
    split_examples = zip(*(torch.split(tensor, split_size) if tensor is not None else [torch.tensor([])]*pieces for tensor in example_tuple))
    # Deepspeed works with a tuple of (features, labels), even if we don't provide a loss_fn to PipelineEngine,
    # and instead compute the loss ourselves in the model. It's okay to just return None for the labels here.
    return [(ex, None) for ex in split_examples]



# Splits an example (feature dict) along the batch dimension into a list of examples.
# Keeping this code because we might want to switch to this way of doing things eventually.
# def split_batch(example, pieces):
#     key, value = example.popitem()
#     input_batch_size = len(value)
#     example[key] = value
#     split_size = input_batch_size // pieces
#     examples = [{} for _ in range(pieces)]
#     for key, value in example.items():
#         assert len(value) == input_batch_size
#         for i, j in enumerate(range(0, input_batch_size, split_size)):
#             examples[i][key] = value[j:j+split_size]
#     return examples


# DataLoader that divides batches into microbatches for gradient accumulation steps when doing
# pipeline parallel training. Iterates indefinitely (deepspeed requirement). Keeps track of epoch.
# Updates epoch as soon as the final batch is returned (notably different from qlora-pipe).
class PipelineDataLoader:
    def __init__(self, dataset, gradient_accumulation_steps, model, num_dataloader_workers=2):
        self.model = model
        self.dataset = dataset
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_dataloader_workers = num_dataloader_workers
        self.iter_called = False
        self.eval_quantile = None
        self.epoch = 1
        self.num_batches_pulled = 0
        self.next_micro_batch = None
        self.recreate_dataloader = False
        # Be careful to only create the DataLoader some bounded number of times: https://github.com/pytorch/pytorch/issues/91252
        self._create_dataloader()
        self.data = self._pull_batches_from_dataloader()

    def reset(self):
        self.epoch = 1
        self.num_batches_pulled = 0
        self.next_micro_batch = None
        self.data = self._pull_batches_from_dataloader()

    def set_eval_quantile(self, quantile):
        self.eval_quantile = quantile

    def __iter__(self):
        self.iter_called = True
        return self

    def __len__(self):
        return len(self.dataset) * self.gradient_accumulation_steps

    def __next__(self):
        if self.next_micro_batch == None:
            self.next_micro_batch = next(self.data)
        ret = self.next_micro_batch
        try:
            self.next_micro_batch = next(self.data)
        except StopIteration:
            if self.recreate_dataloader:
                self._create_dataloader()
                self.recreate_dataloader = False
            self.data = self._pull_batches_from_dataloader()
            self.num_batches_pulled = 0
            self.next_micro_batch = next(self.data)
            self.epoch += 1
        return ret

    def _create_dataloader(self, skip_first_n_batches=None):
        if skip_first_n_batches is not None:
            sampler = SkipFirstNSampler(skip_first_n_batches, len(self.dataset))
        else:
            sampler = None
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            pin_memory=True,
            batch_size=None,
            sampler=sampler,
            num_workers=self.num_dataloader_workers,
            persistent_workers=(self.num_dataloader_workers > 0),
        )

    def _pull_batches_from_dataloader(self):
        for batch in self.dataloader:
            batch = self.model.prepare_inputs(batch, timestep_quantile=self.eval_quantile)
            self.num_batches_pulled += 1
            for micro_batch in split_batch(batch, self.gradient_accumulation_steps):
                yield micro_batch

    # Only the first and last stages in the pipeline pull from the dataloader. Parts of the code need
    # to know the epoch, so we synchronize the epoch so the processes that don't use the dataloader
    # know the current epoch.
    def sync_epoch(self):
        process_group = dist.get_world_group()
        result = [None] * dist.get_world_size(process_group)
        torch.distributed.all_gather_object(result, self.epoch, group=process_group)
        max_epoch = -1
        for epoch in result:
            max_epoch = max(epoch, max_epoch)
        self.epoch = max_epoch

    def state_dict(self):
        return {
            'epoch': self.epoch,
            'num_batches_pulled': self.num_batches_pulled,
        }

    def load_state_dict(self, state_dict):
        assert not self.iter_called
        self.epoch = state_dict['epoch']
        # -1 because by preloading the next micro_batch, it's always going to have one more batch
        # pulled than the actual number of batches iterated by the caller.
        self.num_batches_pulled = state_dict['num_batches_pulled'] - 1
        self._create_dataloader(skip_first_n_batches=self.num_batches_pulled)
        self.data = self._pull_batches_from_dataloader()
        # Recreate the dataloader after the first pass so that it won't skip
        # batches again (we only want it to skip batches the first time).
        self.recreate_dataloader = True


class SkipFirstNSampler(torch.utils.data.Sampler):
    def __init__(self, n, dataset_length):
        super().__init__()
        self.n = n
        self.dataset_length = dataset_length

    def __len__(self):
        return self.dataset_length

    def __iter__(self):
        for i in range(self.n, self.dataset_length):
            yield i


if __name__ == '__main__':
    from utils import common
    common.is_main_process = lambda: True
    from contextlib import contextmanager
    @contextmanager
    def _zero_first():
        yield
    common.zero_first = _zero_first

    from utils import dataset as dataset_util
    dataset_util.DEBUG = True

    from models import flux
    model = flux.CustomFluxPipeline.from_pretrained('/data2/imagegen_models/FLUX.1-dev', torch_dtype=torch.bfloat16)
    model.model_config = {'guidance': 1.0, 'dtype': torch.bfloat16}

    import toml
    dataset_manager = dataset_util.DatasetManager(model)
    with open('/home/anon/code/diffusion-pipe-configs/datasets/tiny1.toml') as f:
        dataset_config = toml.load(f)
    train_data = dataset_util.Dataset(dataset_config, model)
    dataset_manager.register(train_data)
    dataset_manager.cache()

    train_data.post_init(data_parallel_rank=0, data_parallel_world_size=1, per_device_batch_size=1, gradient_accumulation_steps=2)
    print(f'Dataset length: {len(train_data)}')

    for item in train_data:
        pass