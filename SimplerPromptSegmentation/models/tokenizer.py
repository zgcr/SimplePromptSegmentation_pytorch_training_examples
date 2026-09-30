import random

import torch
from transformers import AutoProcessor

# Special token
SEG_TOKEN = "<SEG>"
PSTART_TOKEN = "<p>"
PEND_TOKEN = "</p>"
REGION_TOKEN = "<region>"

# Qwen3VLANDQwen3.5 vision tokens
VISION_START = "<|vision_start|>"
VISION_END = "<|vision_end|>"
IMAGE_TOKEN = "<image>"

# Label ignore index
IGNORE_INDEX = -100

# Default number of placeholder tokens each visual prompt region occupies in
# the VLM input sequence. The model's MaskRegionEncoder produces exactly this
# many feature tokens per region, and the model replaces the embeddings of
# these placeholder tokens with those features via a single masked_scatter.
#
# Emitting the placeholders directly at tokenization time (instead of
# expanding a single <region> token inside the model) makes every derived
# tensor (input_ids/attention_mask/labels/cond_ids/seg_ids/mm_token_type_ids
# and the batch padding) automatically correct, and keeps position_ids valid
# without recomputation.
DEFAULT_NUM_REGION_TOKENS = 12

# Separator used to join multiple <SEG> tokens inside one answer.
SEG_SEPARATOR = ", "

########################################################################
# Text prompt segmentation (REFSEG) templates
#
# {phrases} = "<p>{label1}</p>" or "<p>{label1}</p>, <p>{label2}</p>, ..."
# {segs}    = "<SEG>" or "<SEG>, <SEG>, ..."  (one <SEG> per prompt)
#
# Templates are split into singular/plural variants so that the natural
# language stays grammatical when a single conversation asks for multiple
# targets at once.
########################################################################
REFSEG_QUESTION_TEMPLATES = {
    ('english', 'singular'): [
        "Please identify and segment the {phrases} in this image.",
        "Please segment {phrases} in this image.",
        "What is {phrases} in this image? Please output the corresponding segmentation mask.",
        "Can you segment {phrases} in this image? Please generate the segmentation mask.",
        "Could you provide a segmentation mask for the {phrases} in this image? Please provide the segmentation mask.",
        "Where is the {phrases} in this picture? Please output the segmentation mask.",
        "Can you highlight the {phrases} in this image with a segmentation mask? Please output the segmentation mask.",
        "Could you provide a segmentation mask for the {phrases} in this image? Please respond with the segmentation mask.",
        "Where is the {phrases} in this picture? Please output the corresponding segmentation mask.",
    ],
    ('english', 'plural'): [
        "Please identify and segment the following targets in this image: {phrases}.",
        "Please segment {phrases} in this image.",
        "What are {phrases} in this image? Please output the corresponding segmentation masks.",
        "Can you segment {phrases} in this image? Please generate the segmentation masks.",
        "Could you provide segmentation masks for {phrases} in this image? Please provide the segmentation masks.",
        "Where are {phrases} in this picture? Please output the segmentation masks.",
        "Can you highlight {phrases} in this image with segmentation masks? Please output the segmentation masks.",
        "Could you provide segmentation masks for {phrases} in this image? Please respond with the segmentation masks.",
        "Where are {phrases} in this picture? Please output the corresponding segmentation masks.",
    ],
    ('chinese', 'singular'): [
        "请在这张图片中识别并分割{phrases}。",
        "请分割这张图片中的{phrases}。",
        "这张图片中的{phrases}是什么？请输出对应的分割掩码。",
        "你能分割这张图片中的{phrases}吗？请生成分割掩码。",
        "你能提供这张图片中{phrases}的分割掩码吗？请提供分割掩码。",
        "这张图片中的{phrases}在哪里？请输出分割掩码。",
        "你能用分割掩码标注这张图片中的{phrases}吗？请输出分割掩码。",
        "你能提供这张图片中{phrases}的分割掩码吗？请给出分割掩码。",
        "这张图片中的{phrases}在哪里？请输出对应的分割掩码。",
    ],
    ('chinese', 'plural'): [
        "请在这张图片中识别并分割以下目标：{phrases}。",
        "请分割这张图片中的{phrases}。",
        "这张图片中的{phrases}分别是什么？请输出对应的分割掩码。",
        "你能分割这张图片中的{phrases}吗？请分别生成分割掩码。",
        "你能提供这张图片中{phrases}的分割掩码吗？请分别提供分割掩码。",
        "这张图片中的{phrases}分别在哪里？请输出分割掩码。",
        "你能用分割掩码分别标注这张图片中的{phrases}吗？请输出分割掩码。",
        "你能提供这张图片中{phrases}的分割掩码吗？请分别给出分割掩码。",
        "这张图片中的{phrases}分别在哪里？请输出对应的分割掩码。",
    ],
}

REFSEG_ANSWER_TEMPLATES = {
    ('english', 'singular'): [
        "{segs}.",
        "It is {segs}.",
        "Sure, {segs}.",
        "Sure, it is {segs}.",
        "Sure, the segmentation mask is {segs}.",
    ],
    ('english', 'plural'): [
        "{segs}.",
        "They are {segs}.",
        "Sure, {segs}.",
        "Sure, they are {segs}.",
        "Sure, the segmentation masks are {segs}.",
    ],
    ('chinese', 'singular'): [
        "{segs}。",
        "它是{segs}。",
        "好的，{segs}。",
        "好的，它是{segs}。",
        "好的，分割掩码是{segs}。",
    ],
    ('chinese', 'plural'): [
        "{segs}。",
        "它们分别是{segs}。",
        "好的，{segs}。",
        "好的，它们分别是{segs}。",
        "好的，分割掩码分别是{segs}。",
    ],
}

########################################################################
# Visual prompt segmentation (VGDSEG) templates
#
# {regions} = "<p><region>...<region></p>" repeated and joined by ", ",
#             where each <p>...</p> unit contains num_region_tokens copies
#             of <region>.
# {segs}    = "<SEG>" or "<SEG>, <SEG>, ..."  (one <SEG> per region)
########################################################################
VGDSEG_QUESTION_TEMPLATES = {
    ('english', 'singular'): [
        "Can you segment the image based on the following region: {regions}? Please output the corresponding segmentation mask.",
        "Can you generate a segmentation mask for this image based on the specified region: {regions}? Please generate the segmentation mask.",
        "Can you provide a segmentation mask for this image based on this region: {regions}? Please provide the segmentation mask.",
        "Could you create a segmentation mask for this image according to the specified region: {regions}? Please create the segmentation mask.",
        "Could you output a segmentation mask for this image that highlights the following region: {regions}? Please output the segmentation mask.",
        "Could you provide a segmentation mask for this image according to the specified region: {regions}? Please respond with the segmentation mask.",
    ],
    ('english', 'plural'): [
        "Can you segment the image based on the following regions: {regions}? Please output the corresponding segmentation masks.",
        "Can you generate segmentation masks for this image based on the specified regions: {regions}? Please generate the segmentation masks.",
        "Can you provide segmentation masks for this image based on these regions: {regions}? Please provide the segmentation masks.",
        "Could you create segmentation masks for this image according to the specified regions: {regions}? Please create the segmentation masks.",
        "Could you output segmentation masks for this image that highlight the following regions: {regions}? Please output the segmentation masks.",
        "Could you provide segmentation masks for this image according to the specified regions: {regions}? Please respond with the segmentation masks.",
    ],
    ('chinese', 'singular'): [
        "你能根据以下区域分割这张图片吗：{regions}？请输出对应的分割掩码。",
        "你能根据指定的区域为这张图片生成分割掩码吗：{regions}？请生成分割掩码。",
        "你能根据这个区域为这张图片提供分割掩码吗：{regions}？请提供分割掩码。",
        "你能按照指定的区域为这张图片创建分割掩码吗：{regions}？请创建分割掩码。",
        "你能输出这张图片中以下区域的分割掩码吗：{regions}？请输出分割掩码。",
        "你能按照指定的区域为这张图片提供分割掩码吗：{regions}？请给出分割掩码。",
    ],
    ('chinese', 'plural'): [
        "你能根据以下区域分割这张图片吗：{regions}？请输出对应的分割掩码。",
        "你能根据指定的区域为这张图片分别生成分割掩码吗：{regions}？请生成分割掩码。",
        "你能根据这些区域为这张图片提供分割掩码吗：{regions}？请分别提供分割掩码。",
        "你能按照指定的区域为这张图片分别创建分割掩码吗：{regions}？请创建分割掩码。",
        "你能输出这张图片中以下区域的分割掩码吗：{regions}？请分别输出分割掩码。",
        "你能按照指定的区域为这张图片分别提供分割掩码吗：{regions}？请给出分割掩码。",
    ],
}

VGDSEG_ANSWER_TEMPLATES = {
    ('english', 'singular'): [
        "{segs}.",
        "It is {segs}.",
        "Sure, {segs}.",
        "Sure, it is {segs}.",
        "Sure, the segmentation result is {segs}.",
    ],
    ('english', 'plural'): [
        "{segs}.",
        "They are {segs}.",
        "Sure, {segs}.",
        "Sure, they are {segs}.",
        "Sure, the segmentation results are {segs}.",
    ],
    ('chinese', 'singular'): [
        "{segs}。",
        "它是{segs}。",
        "好的，{segs}。",
        "好的，它是{segs}。",
        "好的，分割结果是{segs}。",
    ],
    ('chinese', 'plural'): [
        "{segs}。",
        "它们分别是{segs}。",
        "好的，{segs}。",
        "好的，它们分别是{segs}。",
        "好的，分割结果分别是{segs}。",
    ],
}


def pick_templates(template_dict, language, num_prompts):
    """Select the singular or plural template pool for a language."""
    plurality = 'singular' if num_prompts == 1 else 'plural'

    return template_dict[(language, plurality)]


class Qwen3VLSegTokenizer:
    """Qwen3VL tokenizer for prompt segmentation.

    Two behaviours differ from the original implementation:

    1. One ``<SEG>`` token is emitted per prompt (per text phrase for REFSEG,
       per region for VGDSEG) instead of a single ``<SEG>`` for the whole
       conversation. This lets the segmentation decoder receive an
       independent conditioning signal for every requested target.

    2. Every visual prompt region is written as ``num_region_tokens`` copies
       of ``<region>`` wrapped in ``<p>...</p>``. The model then only needs a
       single ``masked_scatter`` to inject region features, exactly like it
       already does for image tokens.
    """

    def __init__(self,
                 vlm_model_path,
                 special_tokens=[
                     SEG_TOKEN,
                     PSTART_TOKEN,
                     PEND_TOKEN,
                     REGION_TOKEN,
                 ],
                 num_region_tokens=DEFAULT_NUM_REGION_TOKENS):
        self.processor = AutoProcessor.from_pretrained(vlm_model_path,
                                                       trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

        num_added = self.tokenizer.add_tokens(special_tokens,
                                              special_tokens=True)
        if num_added > 0:
            print(f"{vlm_model_path}: added {num_added} special tokens")

        # Cache special token ids
        self.seg_token_id = self.tokenizer.convert_tokens_to_ids(SEG_TOKEN)
        self.pstart_token_id = self.tokenizer.convert_tokens_to_ids(
            PSTART_TOKEN)
        self.pend_token_id = self.tokenizer.convert_tokens_to_ids(PEND_TOKEN)
        self.region_token_id = self.tokenizer.convert_tokens_to_ids(
            REGION_TOKEN)

        assert int(num_region_tokens) >= 1, 'num_region_tokens must be >= 1'
        self.num_region_tokens = int(num_region_tokens)
        # "<p><region><region>...<region></p>"
        self.region_unit_text = (PSTART_TOKEN +
                                 REGION_TOKEN * self.num_region_tokens +
                                 PEND_TOKEN)

        # Image token format for Qwen3VL
        self.image_token_format = f"{VISION_START}{IMAGE_TOKEN}{VISION_END}"

        # Cache the assistant header token ids for label masking.
        # In Qwen3VL chat template, assistant turn starts with:
        #   "<|im_start|>assistant\n"
        # We tokenize this string to identify where the assistant response
        # begins in the full input_ids sequence.
        self.assistant_header_ids = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False)

        # The assistant turn ends with "<|im_end|>"
        self.im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        self.vocab_size = len(self.tokenizer)

        print(f'vocab_size: {self.vocab_size}, '
              f'num_region_tokens: {self.num_region_tokens}')

    def build_conversations(self, prompt_texts, sample_type, prompt_languages):
        """Build question/answer text from origin text prompts.

        For text prompt (REFSEG):
            ``prompt_texts[i]`` is a list of label strings. Each label is
            wrapped as ``<p>{label}</p>`` and joined by ", ".

        For visual prompt (VGDSEG):
            ``prompt_texts[i]`` is "<region>" repeated and joined by ", ".
            Each region becomes ``<p><region>*T</p>``.

        In both cases the answer contains exactly ``n`` ``<SEG>`` tokens,
        where ``n`` is the number of prompts in that sample, so that
        ``compute_seg_ids`` assigns seg ids 0..n-1 that line up with the
        cond ids produced by ``compute_cond_ids``.

        Args:
            prompt_texts: origin text prompts.
                For text task: list of list of str.
                For visual task: list of str ("<region>, <region>").
            sample_type: str, "text" or "visual".
            prompt_languages: list of str, "chinese" or "english".

        Returns:
            question_texts: list of str
            answer_texts: list of str
            num_prompts: list of int, number of prompts per sample
        """
        B = len(prompt_texts)
        question_texts = []
        answer_texts = []
        num_prompts = []

        for i in range(B):
            language = prompt_languages[i]

            assert language in ['chinese',
                                'english'], f"Invalid language: {language}"

            if sample_type == 'text':
                # Text prompt: prompt_texts[i] is a list of label strings
                labels = prompt_texts[i]
                n = len(labels)
                assert n >= 1, 'text prompt must have at least one label'

                phrases_str = ", ".join(
                    [PSTART_TOKEN + label + PEND_TOKEN for label in labels])

                q_templates = pick_templates(REFSEG_QUESTION_TEMPLATES,
                                             language, n)
                a_templates = pick_templates(REFSEG_ANSWER_TEMPLATES, language,
                                             n)
                question_text = random.choice(q_templates).format(
                    phrases=phrases_str)

            elif sample_type == 'visual':
                # Visual prompt: split "<region>, <region>" and rebuild with
                # num_region_tokens placeholders per region.
                text_regions = prompt_texts[i].split(", ")
                assert all(r == REGION_TOKEN for r in text_regions), (
                    f"Visual prompt must consist of '{REGION_TOKEN}' tokens "
                    f"separated by ', ', but got: {text_regions}")
                n = len(text_regions)
                assert n >= 1, 'visual prompt must have at least one region'

                regions_str = ", ".join([self.region_unit_text] * n)

                q_templates = pick_templates(VGDSEG_QUESTION_TEMPLATES,
                                             language, n)
                a_templates = pick_templates(VGDSEG_ANSWER_TEMPLATES, language,
                                             n)
                question_text = random.choice(q_templates).format(
                    regions=regions_str)

            else:
                raise ValueError(f'Invalid sample_type: {sample_type}')

            # One <SEG> per prompt, joined by ", "
            segs_str = SEG_SEPARATOR.join([SEG_TOKEN] * n)
            answer_text = random.choice(a_templates).format(segs=segs_str)

            question_texts.append(question_text)
            answer_texts.append(answer_text)
            num_prompts.append(n)

        return question_texts, answer_texts, num_prompts

    def build_chat_messages(self, question_text, answer_text, pil_image=None):
        """Build Qwen3VL chat messages format from question/answer text.

        Args:
            question_text: str, the question (may contain <p>, </p>, <region>)
            answer_text: str, the answer (may contain <SEG>)
            pil_image: PIL.Image or None

        Returns:
            list of message dicts for processor.apply_chat_template
        """
        user_content = []
        if pil_image is not None:
            user_content.append({"type": "image", "image": pil_image})
        user_content.append({"type": "text", "text": question_text})

        messages = [
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": answer_text,
                }]
            },
        ]

        return messages

    def encode(self,
               prompt_texts,
               sample_type,
               prompt_languages,
               pil_images=None):
        """Encode a batch of origin text prompts into VLM-ready tensors.

        build_conversations -> build_chat_messages -> tokenize -> masks.

        Args:
            prompt_texts: list of str list, raw text prompts (one per sample).
            sample_type: str, "text" or "visual".
            prompt_languages: list of str, language per sample.
            pil_images: list of PIL.Image or None (one per sample).

        Returns:
            dict with keys:
                input_ids: [B, L] padded token ids
                attention_mask: [B, L] attention mask
                labels: [B, L] labels (IGNORE_INDEX for non-output tokens)
                pixel_values: tensor from VLM image processor, or None
                image_grid_thw: tensor from VLM image processor, or None
                mm_token_type_ids: [B, L] multimodal token type ids
                cond_ids: [B, L] condition ids
                seg_ids: [B, L] segment ids
                region_token_mask: [B, L] boolean mask for <region> positions
                num_prompts: [B] number of prompts per sample
        """
        B = len(prompt_texts)
        if pil_images is None:
            pil_images = [None] * B

        # Step 1: Build question/answer texts from raw prompts
        question_texts, answer_texts, num_prompts = self.build_conversations(
            prompt_texts, sample_type, prompt_languages)

        # Step 2 & 3: Build chat messages, tokenize, extract special masks
        all_input_ids = []
        all_labels = []
        all_cond_ids = []
        all_seg_ids = []
        all_region_masks = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_mm_token_type_ids = []
        for i in range(B):
            messages = self.build_chat_messages(question_texts[i],
                                                answer_texts[i], pil_images[i])

            # Use processor to tokenize and process image
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt")

            # get input_ids:[L]
            input_ids = inputs['input_ids'].squeeze(0)
            all_input_ids.append(input_ids)
            # Compute labels (assistant response has loss, rest is IGNORE)
            all_labels.append(self.compute_labels(input_ids))
            # Compute special token masks
            all_cond_ids.append(self.compute_cond_ids(input_ids))
            all_seg_ids.append(self.compute_seg_ids(input_ids))
            all_region_masks.append(self.compute_region_token_mask(input_ids))

            # Collect pixel_values and image_grid_thw
            if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
                all_pixel_values.append(inputs['pixel_values'])
            if 'image_grid_thw' in inputs and inputs[
                    'image_grid_thw'] is not None:
                all_image_grid_thw.append(inputs['image_grid_thw'])

            # Collect mm_token_type_ids (required by Qwen3VL for M-RoPE)
            if 'mm_token_type_ids' in inputs and inputs[
                    'mm_token_type_ids'] is not None:
                all_mm_token_type_ids.append(
                    inputs['mm_token_type_ids'].squeeze(0))

        # Pad sequences to max length in batch
        max_len = max(ids.shape[0] for ids in all_input_ids)
        pad_token_id = self.tokenizer.pad_token_id or 0

        padded_input_ids = torch.full((B, max_len),
                                      pad_token_id,
                                      dtype=torch.long)
        padded_attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        padded_labels = torch.full((B, max_len),
                                   IGNORE_INDEX,
                                   dtype=torch.long)
        padded_cond_ids = torch.full((B, max_len), -1, dtype=torch.long)
        padded_seg_ids = torch.full((B, max_len), -1, dtype=torch.long)
        padded_region_mask = torch.zeros((B, max_len), dtype=torch.bool)

        # mm_token_type_ids: pad with 0 (text type) for padding positions
        padded_mm_token_type_ids = None
        if len(all_mm_token_type_ids) > 0:
            padded_mm_token_type_ids = torch.zeros((B, max_len),
                                                   dtype=torch.long)

        for i in range(B):
            L = all_input_ids[i].shape[0]
            padded_input_ids[i, :L] = all_input_ids[i]
            padded_attention_mask[i, :L] = 1
            padded_labels[i, :L] = all_labels[i]
            padded_cond_ids[i, :L] = all_cond_ids[i]
            padded_seg_ids[i, :L] = all_seg_ids[i]
            padded_region_mask[i, :L] = all_region_masks[i]
            if padded_mm_token_type_ids is not None and i < len(
                    all_mm_token_type_ids):
                padded_mm_token_type_ids[i, :L] = all_mm_token_type_ids[i]

        # Stack pixel_values and image_grid_thw
        pixel_values = None
        image_grid_thw = None
        if len(all_pixel_values) > 0:
            pixel_values = torch.cat(all_pixel_values, dim=0)
        if len(all_image_grid_thw) > 0:
            image_grid_thw = torch.cat(all_image_grid_thw, dim=0)

        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_mask,
            'labels': padded_labels,
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw,
            'mm_token_type_ids': padded_mm_token_type_ids,
            'cond_ids': padded_cond_ids,
            'seg_ids': padded_seg_ids,
            'region_token_mask': padded_region_mask,
            'num_prompts': torch.tensor(num_prompts, dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=False):
        """Decode a batch of token ids back to text.

        Args:
            token_ids: [B, L] tensor of token ids
            skip_special_tokens: bool, whether to skip special tokens

        Returns:
            list of str, decoded text for each sample in the batch
        """
        decoded_texts = self.tokenizer.batch_decode(
            token_ids, skip_special_tokens=skip_special_tokens)

        return decoded_texts

    def compute_labels(self, input_ids):
        """Compute autoregressive training labels from input_ids.

        Only the assistant's response tokens get their original token ids as
        labels (i.e., they contribute to the loss), while all other tokens
        (system prompt, user message, padding, image placeholders) get
        IGNORE_INDEX=-100 (excluded from the loss).

        Strategy:
        - Find the first "<|im_start|>assistant\\n" header in input_ids
        - The tokens AFTER this header until (and including) "<|im_end|>"
          are the assistant's response -> labels = token ids
        - Everything else -> labels = IGNORE_INDEX

        Because visual prompt region placeholders live inside the *question*,
        they are automatically excluded from the loss without any special
        handling.

        Args:
            input_ids: [L] tensor of token ids

        Returns:
            labels: [L] tensor
        """
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        input_list = input_ids.tolist()
        header_len = len(self.assistant_header_ids)

        # Find the first occurrence of the assistant header
        # (single-turn only: one user message + one assistant response)
        assistant_start = -1
        for i in range(len(input_list) - header_len + 1):
            if input_list[i:i + header_len] == self.assistant_header_ids:
                assistant_start = i + header_len  # position right after header
                break

        # A miss here must be loud. The header is located by an exact token
        # subsequence match, which silently breaks whenever the chat template
        # changes or the tokenizer merges "assistant\n" with the surrounding
        # context. Returning all IGNORE_INDEX in that case would leave the VLM
        # loss with zero supervised tokens, so loss_function() yields NaN and
        # every batch gets skipped (pytorch launcher) or poisons the weights
        # (deepspeed launcher) -- with nothing in the logs pointing at the
        # tokenizer.
        assert assistant_start != -1, (
            f'assistant header {self.assistant_header_ids} not found in '
            f'input_ids; compute_labels would return an all-IGNORE label '
            f'tensor and the VLM loss would be NaN. The chat template of '
            f'this checkpoint probably no longer starts the assistant turn '
            f'with "<|im_start|>assistant\\n".')

        # Find the <|im_end|> token after assistant_start
        assistant_end = len(input_list)  # default: to end
        for i in range(assistant_start, len(input_list)):
            if input_list[i] == self.im_end_id:
                assistant_end = i + 1  # include <|im_end|> itself
                break

        # Set labels for assistant response tokens (including <|im_end|>)
        labels[assistant_start:assistant_end] = input_ids[
            assistant_start:assistant_end]

        # Guard the remaining way to produce an all-IGNORE tensor: the header
        # was found but sits at the very end of the sequence, leaving an empty
        # response span.
        assert bool((labels != IGNORE_INDEX).any()), (
            f'compute_labels produced an all-IGNORE label tensor '
            f'(assistant_start={assistant_start}, assistant_end='
            f'{assistant_end}, seq_len={len(input_list)}); the VLM loss '
            f'would be NaN.')

        return labels

    def compute_cond_ids(self, input_ids):
        """Compute cond_ids from input_ids by finding <p>...</p> ranges.

        Each <p>...</p> range gets a unique cond_id (0, 1, 2, ...).
        Tokens outside any <p>...</p> range get cond_id = -1.
        The <p> and </p> tokens themselves are included in the range.

        For visual prompts this automatically covers all num_region_tokens
        placeholders belonging to a region, so no manual cond id copying is
        needed inside the model any more.

        Args:
            input_ids: [L] tensor of token ids

        Returns:
            cond_ids: [L] tensor of condition ids
        """
        cond_ids = torch.full_like(input_ids, -1)
        pstart_positions = (input_ids == self.pstart_token_id).nonzero(
            as_tuple=True)[0]
        pend_positions = (input_ids == self.pend_token_id).nonzero(
            as_tuple=True)[0]

        # Match each <p> with the nearest following </p>
        cond_idx = 0
        used_pend = set()
        for pstart_pos in pstart_positions:
            # Find the first </p> after this <p> that hasn't been used
            for pend_pos in pend_positions:
                if pend_pos > pstart_pos and pend_pos.item() not in used_pend:
                    # Mark all tokens in [pstart_pos, pend_pos] inclusive
                    cond_ids[pstart_pos:pend_pos + 1] = cond_idx
                    cond_idx += 1
                    used_pend.add(pend_pos.item())
                    break

        return cond_ids

    def compute_seg_ids(self, input_ids):
        """Compute seg_ids from input_ids by finding <SEG> positions.

        Each <SEG> token gets a unique seg_id (0, 1, 2, ...) following its
        order of appearance. Since the answer now emits one <SEG> per prompt
        in the same order as the <p>...</p> phrases/regions of the question,
        seg_id k and cond_id k refer to the same target.

        Args:
            input_ids: [L] tensor of token ids

        Returns:
            seg_ids: [L] tensor of segment ids
        """
        seg_ids = torch.full_like(input_ids, -1)
        seg_positions = (input_ids == self.seg_token_id).nonzero(
            as_tuple=True)[0]
        for i, pos in enumerate(seg_positions):
            seg_ids[pos] = i

        return seg_ids

    def compute_region_token_mask(self, input_ids):
        """Compute boolean mask for <region> token positions.

        Args:
            input_ids: [L] tensor of token ids

        Returns:
            region_mask: [L] boolean tensor, True at <region> positions
        """
        region_mask = (input_ids == self.region_token_id)

        return region_mask


class Qwen35SegTokenizer:
    """Qwen3.5 tokenizer for prompt segmentation.

    Two behaviours differ from the original implementation:

    1. One ``<SEG>`` token is emitted per prompt (per text phrase for REFSEG,
       per region for VGDSEG) instead of a single ``<SEG>`` for the whole
       conversation. This lets the segmentation decoder receive an
       independent conditioning signal for every requested target.

    2. Every visual prompt region is written as ``num_region_tokens`` copies
       of ``<region>`` wrapped in ``<p>...</p>``. The model then only needs a
       single ``masked_scatter`` to inject region features, exactly like it
       already does for image tokens.
    """

    def __init__(self,
                 vlm_model_path,
                 special_tokens=[
                     SEG_TOKEN,
                     PSTART_TOKEN,
                     PEND_TOKEN,
                     REGION_TOKEN,
                 ],
                 num_region_tokens=DEFAULT_NUM_REGION_TOKENS):
        self.processor = AutoProcessor.from_pretrained(vlm_model_path,
                                                       trust_remote_code=True)
        self.tokenizer = self.processor.tokenizer

        num_added = self.tokenizer.add_tokens(special_tokens,
                                              special_tokens=True)
        if num_added > 0:
            print(f"{vlm_model_path}: added {num_added} special tokens")

        # Cache special token ids
        self.seg_token_id = self.tokenizer.convert_tokens_to_ids(SEG_TOKEN)
        self.pstart_token_id = self.tokenizer.convert_tokens_to_ids(
            PSTART_TOKEN)
        self.pend_token_id = self.tokenizer.convert_tokens_to_ids(PEND_TOKEN)
        self.region_token_id = self.tokenizer.convert_tokens_to_ids(
            REGION_TOKEN)

        assert int(num_region_tokens) >= 1, 'num_region_tokens must be >= 1'
        self.num_region_tokens = int(num_region_tokens)
        # "<p><region><region>...<region></p>"
        self.region_unit_text = (PSTART_TOKEN +
                                 REGION_TOKEN * self.num_region_tokens +
                                 PEND_TOKEN)

        # Image token format for Qwen3.5
        self.image_token_format = f"{VISION_START}{IMAGE_TOKEN}{VISION_END}"

        # Cache the assistant header token ids for label masking.
        # In Qwen3.5 chat template, assistant turn starts with:
        #   "<|im_start|>assistant\n"
        # followed by the think block "<think>\n\n</think>\n\n"
        # We tokenize the header string to identify where the assistant
        # response begins in the full input_ids sequence.
        self.assistant_header_ids = self.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False)

        # Cache the think block token ids that Qwen3.5's chat template
        # automatically inserts: "<think>\n\n</think>\n\n" -> 4 tokens
        # These must be skipped when computing labels so they don't
        # participate in the training loss.
        self.think_block_ids = self.tokenizer.encode("<think>\n\n</think>\n\n",
                                                     add_special_tokens=False)

        # The assistant turn ends with "<|im_end|>"
        self.im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")

        self.vocab_size = len(self.tokenizer)

        print(f'vocab_size: {self.vocab_size}, '
              f'num_region_tokens: {self.num_region_tokens}')

    def build_conversations(self, prompt_texts, sample_type, prompt_languages):
        """Build question/answer text from origin text prompts.

        For text prompt (REFSEG):
            ``prompt_texts[i]`` is a list of label strings. Each label is
            wrapped as ``<p>{label}</p>`` and joined by ", ".

        For visual prompt (VGDSEG):
            ``prompt_texts[i]`` is "<region>" repeated and joined by ", ".
            Each region becomes ``<p><region>*T</p>``.

        In both cases the answer contains exactly ``n`` ``<SEG>`` tokens,
        where ``n`` is the number of prompts in that sample, so that
        ``compute_seg_ids`` assigns seg ids 0..n-1 that line up with the
        cond ids produced by ``compute_cond_ids``.

        Args:
            prompt_texts: origin text prompts.
                For text task: list of list of str.
                For visual task: list of str ("<region>, <region>").
            sample_type: str, "text" or "visual".
            prompt_languages: list of str, "chinese" or "english".

        Returns:
            question_texts: list of str
            answer_texts: list of str
            num_prompts: list of int, number of prompts per sample
        """
        B = len(prompt_texts)
        question_texts = []
        answer_texts = []
        num_prompts = []

        for i in range(B):
            language = prompt_languages[i]

            assert language in ['chinese',
                                'english'], f"Invalid language: {language}"

            if sample_type == 'text':
                # Text prompt: prompt_texts[i] is a list of label strings
                labels = prompt_texts[i]
                n = len(labels)
                assert n >= 1, 'text prompt must have at least one label'

                phrases_str = ", ".join(
                    [PSTART_TOKEN + label + PEND_TOKEN for label in labels])

                q_templates = pick_templates(REFSEG_QUESTION_TEMPLATES,
                                             language, n)
                a_templates = pick_templates(REFSEG_ANSWER_TEMPLATES, language,
                                             n)
                question_text = random.choice(q_templates).format(
                    phrases=phrases_str)

            elif sample_type == 'visual':
                # Visual prompt: split "<region>, <region>" and rebuild with
                # num_region_tokens placeholders per region.
                text_regions = prompt_texts[i].split(", ")
                assert all(r == REGION_TOKEN for r in text_regions), (
                    f"Visual prompt must consist of '{REGION_TOKEN}' tokens "
                    f"separated by ', ', but got: {text_regions}")
                n = len(text_regions)
                assert n >= 1, 'visual prompt must have at least one region'

                regions_str = ", ".join([self.region_unit_text] * n)

                q_templates = pick_templates(VGDSEG_QUESTION_TEMPLATES,
                                             language, n)
                a_templates = pick_templates(VGDSEG_ANSWER_TEMPLATES, language,
                                             n)
                question_text = random.choice(q_templates).format(
                    regions=regions_str)

            else:
                raise ValueError(f'Invalid sample_type: {sample_type}')

            # One <SEG> per prompt, joined by ", "
            segs_str = SEG_SEPARATOR.join([SEG_TOKEN] * n)
            answer_text = random.choice(a_templates).format(segs=segs_str)

            question_texts.append(question_text)
            answer_texts.append(answer_text)
            num_prompts.append(n)

        return question_texts, answer_texts, num_prompts

    def build_chat_messages(self, question_text, answer_text, pil_image=None):
        """Build Qwen3.5 chat messages format from question/answer text.

        Args:
            question_text: str, the question (may contain <p>, </p>, <region>)
            answer_text: str, the answer (may contain <SEG>)
            pil_image: PIL.Image or None

        Returns:
            list of message dicts for processor.apply_chat_template
        """
        user_content = []
        if pil_image is not None:
            user_content.append({"type": "image", "image": pil_image})
        user_content.append({"type": "text", "text": question_text})

        messages = [
            {
                "role": "user",
                "content": user_content,
            },
            {
                "role": "assistant",
                "content": [{
                    "type": "text",
                    "text": answer_text,
                }]
            },
        ]

        return messages

    def encode(self,
               prompt_texts,
               sample_type,
               prompt_languages,
               pil_images=None):
        """Encode a batch of origin text prompts into VLM-ready tensors.

        build_conversations -> build_chat_messages -> tokenize -> masks.

        Args:
            prompt_texts: list of str list, raw text prompts (one per sample).
            sample_type: str, "text" or "visual".
            prompt_languages: list of str, language per sample.
            pil_images: list of PIL.Image or None (one per sample).

        Returns:
            dict with keys:
                input_ids: [B, L] padded token ids
                attention_mask: [B, L] attention mask
                labels: [B, L] labels (IGNORE_INDEX for non-output tokens)
                pixel_values: tensor from VLM image processor, or None
                image_grid_thw: tensor from VLM image processor, or None
                mm_token_type_ids: [B, L] multimodal token type ids
                cond_ids: [B, L] condition ids
                seg_ids: [B, L] segment ids
                region_token_mask: [B, L] boolean mask for <region> positions
                num_prompts: [B] number of prompts per sample
        """
        B = len(prompt_texts)
        if pil_images is None:
            pil_images = [None] * B

        # Step 1: Build question/answer texts from raw prompts
        question_texts, answer_texts, num_prompts = self.build_conversations(
            prompt_texts, sample_type, prompt_languages)

        # Step 2 & 3: Build chat messages, tokenize, extract special masks
        all_input_ids = []
        all_labels = []
        all_cond_ids = []
        all_seg_ids = []
        all_region_masks = []
        all_pixel_values = []
        all_image_grid_thw = []
        all_mm_token_type_ids = []
        for i in range(B):
            messages = self.build_chat_messages(question_texts[i],
                                                answer_texts[i], pil_images[i])

            # Use processor to tokenize and process image
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
                return_dict=True,
                return_tensors="pt")

            # get input_ids:[L]
            input_ids = inputs['input_ids'].squeeze(0)
            all_input_ids.append(input_ids)
            # Compute labels (assistant response has loss, rest is IGNORE)
            all_labels.append(self.compute_labels(input_ids))
            # Compute special token masks
            all_cond_ids.append(self.compute_cond_ids(input_ids))
            all_seg_ids.append(self.compute_seg_ids(input_ids))
            all_region_masks.append(self.compute_region_token_mask(input_ids))

            # Collect pixel_values and image_grid_thw
            if 'pixel_values' in inputs and inputs['pixel_values'] is not None:
                all_pixel_values.append(inputs['pixel_values'])
            if 'image_grid_thw' in inputs and inputs[
                    'image_grid_thw'] is not None:
                all_image_grid_thw.append(inputs['image_grid_thw'])

            # Collect mm_token_type_ids (required by Qwen3.5 for M-RoPE)
            if 'mm_token_type_ids' in inputs and inputs[
                    'mm_token_type_ids'] is not None:
                all_mm_token_type_ids.append(
                    inputs['mm_token_type_ids'].squeeze(0))

        # Pad sequences to max length in batch
        max_len = max(ids.shape[0] for ids in all_input_ids)
        pad_token_id = self.tokenizer.pad_token_id or 0

        padded_input_ids = torch.full((B, max_len),
                                      pad_token_id,
                                      dtype=torch.long)
        padded_attention_mask = torch.zeros((B, max_len), dtype=torch.long)
        padded_labels = torch.full((B, max_len),
                                   IGNORE_INDEX,
                                   dtype=torch.long)
        padded_cond_ids = torch.full((B, max_len), -1, dtype=torch.long)
        padded_seg_ids = torch.full((B, max_len), -1, dtype=torch.long)
        padded_region_mask = torch.zeros((B, max_len), dtype=torch.bool)

        # mm_token_type_ids: pad with 0 (text type) for padding positions
        padded_mm_token_type_ids = None
        if len(all_mm_token_type_ids) > 0:
            padded_mm_token_type_ids = torch.zeros((B, max_len),
                                                   dtype=torch.long)

        for i in range(B):
            L = all_input_ids[i].shape[0]
            padded_input_ids[i, :L] = all_input_ids[i]
            padded_attention_mask[i, :L] = 1
            padded_labels[i, :L] = all_labels[i]
            padded_cond_ids[i, :L] = all_cond_ids[i]
            padded_seg_ids[i, :L] = all_seg_ids[i]
            padded_region_mask[i, :L] = all_region_masks[i]
            if padded_mm_token_type_ids is not None and i < len(
                    all_mm_token_type_ids):
                padded_mm_token_type_ids[i, :L] = all_mm_token_type_ids[i]

        # Stack pixel_values and image_grid_thw
        pixel_values = None
        image_grid_thw = None
        if len(all_pixel_values) > 0:
            pixel_values = torch.cat(all_pixel_values, dim=0)
        if len(all_image_grid_thw) > 0:
            image_grid_thw = torch.cat(all_image_grid_thw, dim=0)

        return {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_mask,
            'labels': padded_labels,
            'pixel_values': pixel_values,
            'image_grid_thw': image_grid_thw,
            'mm_token_type_ids': padded_mm_token_type_ids,
            'cond_ids': padded_cond_ids,
            'seg_ids': padded_seg_ids,
            'region_token_mask': padded_region_mask,
            'num_prompts': torch.tensor(num_prompts, dtype=torch.long),
        }

    def decode(self, token_ids, skip_special_tokens=False):
        """Decode a batch of token ids back to text.

        Args:
            token_ids: [B, L] tensor of token ids
            skip_special_tokens: bool, whether to skip special tokens

        Returns:
            list of str, decoded text for each sample in the batch
        """
        decoded_texts = self.tokenizer.batch_decode(
            token_ids, skip_special_tokens=skip_special_tokens)

        return decoded_texts

    def compute_labels(self, input_ids):
        """Compute autoregressive training labels from input_ids.

        Only the assistant's *actual* response tokens get their original
        token ids as labels, while all other tokens (system prompt, user
        message, padding, image placeholders, and the automatically inserted
        ``<think>\\n\\n</think>\\n\\n`` block) get IGNORE_INDEX=-100.

        Strategy:
        - Find the first "<|im_start|>assistant\\n" header in input_ids
        - Skip the 4-token think block "<think>\\n\\n</think>\\n\\n" that
          Qwen3.5's chat template automatically inserts right after the header
        - The tokens AFTER the think block until (and including) "<|im_end|>"
          are the assistant's actual response -> labels = token ids
        - Everything else -> labels = IGNORE_INDEX

        Because visual prompt region placeholders live inside the *question*,
        they are automatically excluded from the loss without any special
        handling.

        Args:
            input_ids: [L] tensor of token ids

        Returns:
            labels: [L] tensor
        """
        labels = torch.full_like(input_ids, IGNORE_INDEX)
        input_list = input_ids.tolist()
        header_len = len(self.assistant_header_ids)
        think_len = len(self.think_block_ids)

        # Find the first occurrence of the assistant header
        # (single-turn only: one user message + one assistant response)
        assistant_start = -1
        for i in range(len(input_list) - header_len + 1):
            if input_list[i:i + header_len] == self.assistant_header_ids:
                # Position right after header
                header_end = i + header_len
                # Verify and skip the think block
                if (header_end + think_len <= len(input_list)
                        and input_list[header_end:header_end + think_len]
                        == self.think_block_ids):
                    # Skip past the think block to the actual response
                    assistant_start = header_end + think_len
                else:
                    # Fallback: no think block found (shouldn't happen with
                    # Qwen3.5 template, but handle gracefully)
                    assistant_start = header_end
                break

        # A miss here must be loud. The header is located by an exact token
        # subsequence match, which silently breaks whenever the chat template
        # changes or the tokenizer merges "assistant\n" with the surrounding
        # context. Returning all IGNORE_INDEX in that case would leave the VLM
        # loss with zero supervised tokens, so loss_function() yields NaN and
        # every batch gets skipped (pytorch launcher) or poisons the weights
        # (deepspeed launcher) -- with nothing in the logs pointing at the
        # tokenizer.
        assert assistant_start != -1, (
            f'assistant header {self.assistant_header_ids} not found in '
            f'input_ids; compute_labels would return an all-IGNORE label '
            f'tensor and the VLM loss would be NaN. The chat template of '
            f'this checkpoint probably no longer starts the assistant turn '
            f'with "<|im_start|>assistant\\n".')

        # Find the <|im_end|> token after assistant_start
        assistant_end = len(input_list)  # default: to end
        for i in range(assistant_start, len(input_list)):
            if input_list[i] == self.im_end_id:
                assistant_end = i + 1  # include <|im_end|> itself
                break

        # Set labels for assistant response tokens (including <|im_end|>)
        labels[assistant_start:assistant_end] = input_ids[
            assistant_start:assistant_end]

        # Guard the remaining ways to produce an all-IGNORE tensor: the header
        # (plus the think block that Qwen3.5 appends to it) can consume the
        # whole sequence, leaving an empty response span.
        assert bool((labels != IGNORE_INDEX).any()), (
            f'compute_labels produced an all-IGNORE label tensor '
            f'(assistant_start={assistant_start}, assistant_end='
            f'{assistant_end}, seq_len={len(input_list)}); the VLM loss '
            f'would be NaN.')

        return labels

    def compute_cond_ids(self, input_ids):
        """Compute cond_ids from input_ids by finding <p>...</p> ranges.

        Each <p>...</p> range gets a unique cond_id (0, 1, 2, ...).
        Tokens outside any <p>...</p> range get cond_id = -1.
        The <p> and </p> tokens themselves are included in the range.

        For visual prompts this automatically covers all num_region_tokens
        placeholders belonging to a region, so no manual cond id copying is
        needed inside the model any more.

        Args:
            input_ids: [L] tensor of token ids

        Returns:
            cond_ids: [L] tensor of condition ids
        """
        cond_ids = torch.full_like(input_ids, -1)
        pstart_positions = (input_ids == self.pstart_token_id).nonzero(
            as_tuple=True)[0]
        pend_positions = (input_ids == self.pend_token_id).nonzero(
            as_tuple=True)[0]

        # Match each <p> with the nearest following </p>
        cond_idx = 0
        used_pend = set()
        for pstart_pos in pstart_positions:
            # Find the first </p> after this <p> that hasn't been used
            for pend_pos in pend_positions:
                if pend_pos > pstart_pos and pend_pos.item() not in used_pend:
                    # Mark all tokens in [pstart_pos, pend_pos] inclusive
                    cond_ids[pstart_pos:pend_pos + 1] = cond_idx
                    cond_idx += 1
                    used_pend.add(pend_pos.item())
                    break

        return cond_ids

    def compute_seg_ids(self, input_ids):
        """Compute seg_ids from input_ids by finding <SEG> positions.

        Each <SEG> token gets a unique seg_id (0, 1, 2, ...) following its
        order of appearance. Since the answer now emits one <SEG> per prompt
        in the same order as the <p>...</p> phrases/regions of the question,
        seg_id k and cond_id k refer to the same target.

        Args:
            input_ids: [L] tensor of token ids

        Returns:
            seg_ids: [L] tensor of segment ids
        """
        seg_ids = torch.full_like(input_ids, -1)
        seg_positions = (input_ids == self.seg_token_id).nonzero(
            as_tuple=True)[0]
        for i, pos in enumerate(seg_positions):
            seg_ids[pos] = i

        return seg_ids

    def compute_region_token_mask(self, input_ids):
        """Compute boolean mask for <region> token positions.

        Args:
            input_ids: [L] tensor of token ids

        Returns:
            region_mask: [L] boolean tensor, True at <region> positions
        """
        region_mask = (input_ids == self.region_token_id)

        return region_mask
