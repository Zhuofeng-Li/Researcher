from ai_researcher.utils import get_reviewer_score
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class CycleReviewer:
    """
    A class for evaluating research papers using CycleReviewer models.
    """

    def __init__(self,
                 model_name="WestlakeNLP/CycleReviewer-ML-Llama-3.1-8B",
                 device="cuda",
                 tensor_parallel_size=1,
                 gpu_memory_utilization=0.95):
        """
        Initialize the CycleReviewer.

        Args:
            model_name (str): Model name to use. Default is "WestlakeNLP/CycleReviewer-ML-Llama-3.1-8B"
            device (str): Device to run the model on. Default is "cuda"
            tensor_parallel_size (int): Number of GPUs to use for tensor parallelism
            gpu_memory_utilization (float): Fraction of GPU memory to use
        """

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model using vLLM
        self.model = LLM(
            model=model_name,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=50000,
            gpu_memory_utilization=gpu_memory_utilization
        )

        # Store model configuration for reference
        self.model_name = model_name
        self.model_config = {
            "tensor_parallel_size": tensor_parallel_size,
            "gpu_memory_utilization": gpu_memory_utilization
        }

    def evaluate(self, paper_context, include_detailed_feedback=True):
        """
        Evaluate a research paper.

        Args:
            paper_context (list or str): Paper to be reviewed
            include_detailed_feedback (bool): Whether to include detailed review sections

        Returns:
            dict: Review of the paper with various components
        """
        # Prepare system prompt for review
        system_prompt = \
            """You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers. For each paper submitted, conduct a comprehensive review addressing the following aspects:
    
            1. Summary: Briefly outline main points and objectives.
            2. Soundness: Assess methodology and logical consistency.
            3. Presentation: Evaluate clarity, organization, and visual aids.
            4. Contribution: Analyze significance and novelty in the field.
            5. Strengths: Identify the paper's strongest aspects.
            6. Weaknesses: Point out areas for improvement.
            7. Questions: Pose questions for the authors.
            8. Rating: Score 1-10, justify your rating.
            9. Meta Review: Provide overall assessment and recommendation (Accept/Reject).
    
            Maintain objectivity and provide specific examples from the paper to support your evaluation.
    
            You need to fill out **4** review opinions."""

        # Prepare paper context
        if type(paper_context) == str:
            paper_context = [paper_context]



        generated_reviews = []
        batch_size = 10
        for n in range(0,len(paper_context),batch_size):
            # Apply chat template
            prompts = []
            for r in range(min(batch_size, len(paper_context) - n)):
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": paper_context[r+n]}
                ]
                input_text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                prompts.append(input_text)
            # Prepare sampling parameters
            sampling_params = SamplingParams(
                temperature=0.4,
                top_p=0.95,
                max_tokens=7000
            )

            # Generate review
            outputs = self.model.generate(
                prompts,
                sampling_params
            )

            # Process generated review text
            for output_num in range(len(outputs)):
                # Process generated text
                generated_text = outputs[output_num].outputs[0].text
                # Use existing CycleResearcher utility to parse generated text
                review = get_reviewer_score(generated_text)
                generated_reviews.append(review)

        return generated_reviews
