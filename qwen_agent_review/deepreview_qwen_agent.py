import re
import requests
from qwen_agent.agents import Assistant


# Helper Functions for Best Mode
# (Reused from deep_reviewer.py)

def extract_questions_from_content(content: str) -> list[str]:
    """Extract questions from the questions block (e.g., \boxed_questions{...})."""
    questions = []
    # Attempt to find the content within oxed_questions{}
    # This regex is a common way to find such blocks if they exist.
    # If the questions are simply listed after a header like "❓ Questions",
    # this part might need adjustment based on actual LLM output format.

    # First, try to find a specific block like oxed_questions{}
    boxed_questions_match = re.search(r'\boxed_questions\{(.*?)\}', content, re.DOTALL)
    lines = [] # Initialize lines to an empty list
    if boxed_questions_match:
        questions_block = boxed_questions_match.group(1)
        # Assuming questions within the block are separated by newlines
        lines = [line.strip() for line in questions_block.split('\n') if line.strip()]
    else:
        # Fallback or alternative: if questions are under a "## Questions" or "❓ Questions" header
        # This part might need refinement based on the actual output format from the LLM.
        # For now, let's assume questions are separated by newlines after such a header.
        if "❓ Questions" in content: # Or a similar marker
            potential_questions_section = content.split("❓ Questions", 1)[-1]
            lines = [line.strip() for line in potential_questions_section.split('\n') if line.strip()]
        elif "## Questions" in content: # Handle markdown style headers
            potential_questions_section = content.split("## Questions", 1)[-1]
            lines = [line.strip() for line in potential_questions_section.split('\n') if line.strip()]
        else: # if no specific block found, assume content itself might be questions or needs different parsing.
            # This part needs to be robust. For now, using the provided logic from main.py's extract_questions_from_content
            # This assumes questions are separated by newlines.
            lines = [line.strip() for line in content.split('\n') if line.strip()]

    # Process lines to extract actual questions
    for line in lines:
        # Skip lines that are not questions (headers, etc.)
        # The flask example had:
        # if line.startswith('#') or not line:
        #    continue
        # This might need to be adapted if the LLM output for questions is different.
        # For now, let's assume any non-empty line in this block is a question.
        # A more robust solution might look for lines ending with '?' or starting with a number/bullet.
        cleaned_line = line.lstrip("0123456789. ").strip() # Remove leading numbers/bullets
        if cleaned_line and cleaned_line != "}": # Ensure it's not just the closing brace of a block
            questions.append(cleaned_line)

    # Deduplicate questions
    return list(dict.fromkeys(questions))


def retrieve_information(questions: list[str]) -> list[dict]:
    """Retrieve information for questions using the OpenScholar external API."""
    if not questions:
        return []
    try:
        # The URL for the OpenScholar API
        openscholar_api_url = 'http://127.0.0.1:38015/batch_ask'
        response = requests.post(
            openscholar_api_url,
            json={"questions": questions},
            timeout=600  # Set a reasonable timeout (in seconds)
        )

        if response.status_code == 200:
            # Assuming the API returns a JSON with a 'results' key
            # which is a list of dictionaries, one for each question.
            return response.json().get('results', [])
        else:
            # Log error or handle appropriately
            print(f"Error retrieving information from OpenScholar API: {response.status_code} - {response.text}")
            return [{"error": f"API Error {response.status_code}", "output": "", "final_passages": ""} for _ in questions]
    except requests.exceptions.RequestException as e:
        # Handle network errors, timeouts, etc.
        print(f"Exception during information retrieval: {str(e)}")
        return [{"error": f"RequestException: {str(e)}", "output": "", "final_passages": ""} for _ in questions]


def get_question_and_answer_text(questions: list[str], results: list[dict]) -> str:
    """Format questions and answers for the second model call."""
    qa_text_parts = []
    for i, question in enumerate(questions):
        qa_text_parts.append(f"## Question {i + 1}:\n{question}")
        if i < len(results) and results[i]:
            result = results[i]
            passages = result.get("final_passages", "N/A")
            answer = result.get("output", "N/A")
            # Sanitize content slightly for inclusion in a prompt if necessary, though LLMs are usually robust.
            # The flask app used .replace('"', "'").replace('\\', '') which might be too aggressive.
            # Keeping it simple here.
            qa_text_parts.append(f"### Retrieved Passages:\n{passages}")
            qa_text_parts.append(f"### Answer from OpenScholar:\n{answer}")
        else:
            qa_text_parts.append("### Retrieved Passages:\nNo information retrieved.")
            qa_text_parts.append("### Answer from OpenScholar:\nNo answer retrieved.")
        qa_text_parts.append("**********") # Separator

    return "\n\n".join(qa_text_parts)


class DeepReviewerQwenAgent:
    """
    A class for generating automated academic peer reviews using Qwen Agent with DashScope API.
    This implementation uses Qwen Agent for the Best Mode logic.
    """

    def __init__(self,
                 api_key=None,
                 model='qwen3-235b-a22b',
                 enable_thinking=False):
        """
        Initialize the DeepReviewerQwenAgent.

        Args:
            api_key (str): DashScope API key. If None, will try to get from environment.
            model (str): Model name to use with DashScope
            enable_thinking (bool): Whether to enable thinking mode in the model
        """
        # Set up LLM configuration
        self.llm_cfg = {
            'model': model,
            'model_type': 'qwen_dashscope',
            'generate_cfg': {
                'enable_thinking': enable_thinking,
            },
        }

        # Add API key if provided
        if api_key:
            self.llm_cfg['api_key'] = api_key

        # Initialize the Qwen Agent Assistant
        self.bot = Assistant(llm=self.llm_cfg)

    def _generate_system_prompt(self, reviewer_num=4):
        """
        Generate the system prompt for Best Mode review.

        Args:
            reviewer_num (int): Number of reviewers to simulate

        Returns:
            str: System prompt for Best Mode
        """
        simreviewer_prompt = "When you simulate different reviewers, write the sections in this order: Summary, Soundness, Presentation, Contribution, Strengths, Weaknesses, Suggestions, Questions, Rating and Confidence."

        prompt = f"""You are an expert academic reviewer tasked with providing a thorough and balanced evaluation of research papers. Your thinking mode is Best Mode. In this mode, you should aim to provide the most reliable review results by conducting a thorough analysis of the paper. I allow you to use search tools to obtain background knowledge about the paper - please provide three different questions. I will help you with the search. After you complete your thinking, you should review by simulating {reviewer_num} different reviewers, and use self-verification to double-check any paper deficiencies identified. Finally, provide complete review results."""
        return prompt + simreviewer_prompt

    def evaluate_best_mode(self, paper_context, reviewer_num=4):
        """
        Generate a peer review using Best Mode with Qwen Agent.

        This method implements a two-step process:
        1. First LLM call to generate initial review with questions
        2. Retrieve information for questions and make second LLM call with context

        Args:
            paper_context (str): The paper content to review. Can be a single string or a list.
            reviewer_num (int): Number of reviewers to simulate

        Returns:
            list: A list of structured reviews (dictionaries).
        """
        system_prompt = self._generate_system_prompt(reviewer_num=reviewer_num)

        if isinstance(paper_context, str):
            paper_contexts = [paper_context]
        elif isinstance(paper_context, list):
            paper_contexts = paper_context
        else:
            raise TypeError("paper_context must be a string or a list of strings.")

        generated_reviews_batch = []

        # Process papers one by one (Best Mode is sequential)
        for single_paper_context in paper_contexts:
            # --- Step 1: First LLM Call (Best Mode) ---
            print(f"Step 1: Generating initial review with questions...")
            messages_step1 = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': single_paper_context}
            ]

            # Run the agent
            responses_step1 = None
            for responses in self.bot.run(messages=messages_step1):
                responses_step1 = responses

            # Extract the generated text from the response
            if not responses_step1:
                print("Warning: No response from first LLM call")
                continue

            # Get the assistant's response
            generated_text_step1 = ""
            for response in responses_step1:
                if response.get('role') == 'assistant':
                    generated_text_step1 += response.get('content', '')

            if not generated_text_step1:
                print("Warning: Empty response from first LLM call")
                continue

            print(f"Step 1 completed. Generated text length: {len(generated_text_step1)}")

            # # --- Step 2: Extract Questions (Best Mode) ---
            # questions = extract_questions_from_content(generated_text_step1)
            # print(f"Extracted {len(questions)} questions")

            # if not questions:
            #     # Fallback: parse the step 1 output as the final review
            #     print("No questions found, using first step output as final review")
            #     review = self._parse_review(generated_text_step1)
            #     generated_reviews_batch.append(review)
            #     continue

            # # --- Step 3: Retrieve Information from OpenScholar (Best Mode) ---
            # print(f"Step 3: Retrieving information for questions...")
            # retrieved_data = retrieve_information(questions)

            # # --- Step 4: Format Q&A Text (Best Mode) ---
            # qa_text = get_question_and_answer_text(questions, retrieved_data)
            # print(f"Formatted Q&A text length: {len(qa_text)}")

            # # --- Step 5: Second LLM Call (Best Mode) ---
            # print(f"Step 5: Generating final review with retrieved information...")

            # # Create a new agent instance for the second call
            # bot2 = Assistant(llm=self.llm_cfg)

            # messages_step2 = [
            #     {'role': 'system', 'content': system_prompt},
            #     {'role': 'user', 'content': single_paper_context},
            #     {'role': 'assistant', 'content': generated_text_step1},
            #     {'role': 'user', 'content': f"Here is the retrieved information for your questions:\n\n{qa_text}\n\nNow please provide the final comprehensive review."}
            # ]

            # # Run the agent for second call
            # responses_step2 = None
            # for responses in bot2.run(messages=messages_step2):
            #     responses_step2 = responses

            # # Extract the generated text from the second response
            # if not responses_step2:
            #     print("Warning: No response from second LLM call, using first step output")
            #     review = self._parse_review(generated_text_step1)
            #     generated_reviews_batch.append(review)
            #     continue

            # # Get the assistant's response
            # generated_text_step2 = ""
            # for response in responses_step2:
            #     if response.get('role') == 'assistant':
            #         generated_text_step2 += response.get('content', '')

            # if not generated_text_step2:
            #     print("Warning: Empty response from second LLM call, using first step output")
            #     review = self._parse_review(generated_text_step1)
            #     generated_reviews_batch.append(review)
            #     continue

            # print(f"Step 5 completed. Final review length: {len(generated_text_step2)}")

            # # Parse the final review
            # review = self._parse_review(generated_text_step2)
            generated_reviews_batch.append(generated_text_step1)

        return generated_reviews_batch

    def _parse_review(self, generated_text):
        """
        Parse the generated review text into structured format.

        Args:
            generated_text (str): Raw generated review text

        Returns:
            dict: Structured review with metadata and reviews
        """
        result = {
            "raw_text": generated_text,
            "reviews": [],
            "meta_review": {},
            "decision": ""
        }

        # Extract meta review if present
        meta_review_match = re.search(r'\\boxed_review\{(.*?)\n}', generated_text, re.DOTALL)
        if meta_review_match:
            result["meta_review"]['content'] = meta_review_match.group(1).strip()
            section = meta_review_match.group(1).strip()
            # Extract summary
            summary_match = re.search(r'## Summary:\s+(.*?)(?=##|\Z)', section, re.DOTALL)
            if summary_match:
                result["meta_review"]["summary"] = summary_match.group(1).strip()

            # Extract rating
            rating_match = re.search(r'## Rating:\s+(.*?)(?=##|\Z)', section, re.DOTALL)
            if rating_match:
                rating_text = rating_match.group(1).strip()
                # Try to extract a numerical rating (1-10)
                number_match = re.search(r'(\d+(?:\.\d+)?)', rating_text)
                if number_match:
                    result["meta_review"]["rating"] = float(number_match.group(1))
                else:
                    result["meta_review"]["rating"] = rating_text

            # Extract other sections as needed
            for section_name in ["Soundness", "Presentation", "Contribution",
                                 "Strengths", "Weaknesses", "Suggestions", "Questions"]:
                section_match = re.search(f'## {section_name}:\s+(.*?)(?=##|\Z)', section, re.DOTALL)
                if section_match:
                    result["meta_review"][section_name.lower()] = section_match.group(1).strip()

        # Extract simulated reviewers' feedback
        simreviewer_match = re.search(r'\\boxed_simreviewers\{(.*?)\n}', generated_text, re.DOTALL)
        if simreviewer_match:
            simreviewer_text = simreviewer_match.group(1).strip()
            # Split into individual reviewer sections
            reviewer_sections = re.split(r'## Reviewer \d+', simreviewer_text)
            # Skip the first empty section if it exists
            if reviewer_sections and not reviewer_sections[0].strip():
                reviewer_sections = reviewer_sections[1:]

            for i, section in enumerate(reviewer_sections):
                review = {
                    "reviewer_id": i + 1,
                    "text": section.strip()
                }

                # Extract summary
                summary_match = re.search(r'## Summary:\s+(.*?)(?=##|\Z)', section, re.DOTALL)
                if summary_match:
                    review["summary"] = summary_match.group(1).strip()

                # Extract rating
                rating_match = re.search(r'## Rating:\s+(.*?)(?=##|\Z)', section, re.DOTALL)
                if rating_match:
                    rating_text = rating_match.group(1).strip()
                    # Try to extract a numerical rating (1-10)
                    number_match = re.search(r'(\d+(?:\.\d+)?)', rating_text)
                    if number_match:
                        review["rating"] = float(number_match.group(1))
                    else:
                        review["rating"] = rating_text

                # Extract other sections as needed
                for section_name in ["Soundness", "Presentation", "Contribution",
                                     "Strengths", "Weaknesses", "Suggestions", "Questions"]:
                    section_match = re.search(f'## {section_name}:\s+(.*?)(?=##|\Z)', section, re.DOTALL)
                    if section_match:
                        review[section_name.lower()] = section_match.group(1).strip()

                result["reviews"].append(review)

        # Extract decision if present
        decision_match = re.search(r'## Decision:\s*\n\s*(\w+)', generated_text)
        if decision_match:
            result["decision"] = decision_match.group(1).strip()

        return result


# Example usage
if __name__ == "__main__":
    # Initialize the reviewer with Qwen Agent
    reviewer = DeepReviewerQwenAgent(
        api_key='sk-2229323f311843b5b44c9d96ee86ccd5',
        model='qwen3-8b',
        enable_thinking=False
    )

    # Example paper context 
    paper_context = "\\title{Spectral learning of shared dynamics between generalized-linear processes}\n\n\\begin{abstract}\nAcross various science and engineering applications, there often arises a need to predict the dynamics of one data stream from another. Further, these data streams may have different statistical properties. Studying the dynamical relationship between such processes, especially for the purpose of predicting one from the other, requires accounting for their distinct statistics while also dissociating their shared dynamical subspace. Existing analytical modeling approaches, however, do not address both of these needs. Here we propose a path forward by deriving a novel analytical multi-step subspace identification algorithm that can learn a model for a primary generalized-linear process (called ``predictor\"), while also dissociating the dynamics shared with a secondary process. We demonstrate a specific application of our approach for modeling discrete Poisson point-processes activity, while finding the dynamics shared with continuous Gaussian processes. In simulations, we show that our algorithm accurately prioritizes identification of shared dynamics. Further, we also demonstrate that the method can additionally model the disjoint dynamics that exist only in the predictor Poisson data stream, if desired. Similarly, we apply our algorithm on a biological dataset to learn models of dynamics in Poisson neural population spiking streams that predict dynamics in movement streams. Compared with existing Poisson subspace identification methods, models learned with our method decoded movements better and with lower-dimensional latent states. Lastly, we discuss regimes in which our assumptions might not be met and provide recommendations and possible future directions of investigation.\n\\end{abstract}\n\n\\section{1 INTRODUCTION  }\n\n\nModeling the shared dynamics between temporally-structured observations with different statistical properties is useful across multiple application domains, including neuroscience and biomedical engineering (D’mello & Kory, 2015; Lu et al., 2021). However, building models of the dynamic relation between such signals is challenging for two key reasons. First, continuous- and discretevalued observations exhibit different statistics, which the modeling approach must appropriately reconcile. Second, residual (i.e., unshared or unique) dynamics present in each observation stream can obscure and confound modeling of their shared dynamics (Allen et al., 2019; Stringer et al., 2019; Sani et al., 2021). Thus, the modeling approach also needs a way to accurately dissociate and prioritize identification of the shared dynamics. Current analytical methods do not simultaneously enable both of these capabilities, which is what we address here.  \n\nLinear dynamical state-space models (SSMs) are a commonly used framework for modeling dynamics using a low-dimensional latent variable that evolves over time (Paninski et al., 2010; Macke et al., 2015; Newman et al., 2023). Even though the past decade has seen an increased use of artificial neural networks and deep learning methods for training dynamical models of time-series data (Pandarinath et al., 2018; Hurwitz et al., 2021; Kramer et al., 2022; Schneider et al., 2023), analytical SSMs still remain widely popular due to their interpretability and broad applicability both in scientific investigations and in real-time engineering applications (Kao et al., 2015; Aghagolzadeh & Truccolo, 2016; Lu et al., 2021; Yang et al., 2021; Newman et al., 2023). For Gauss-Markov models with continuous Gaussian observations, subspace system identification (SSID) theory provides computationally efficient non-iterative algorithms for analytically learning state-space models, both with and without identification of shared dynamics and dissociation of intrinsic vs input-driven activity(Van Overschee & De Moor, 1996; Katayama, 2005; Sani et al., 2021; Galgali et al., 2023; Vahidi et al., 2023). These methods, however, are not applicable to generalized-linear processes with non-Gaussian observations. While there has been work extending SSID to generalized-linear processes, such as Poisson and Bernoulli observations (Buesing et al., 2012; Stone et al., 2023), these methods only learn the dynamics of a single observation time-series rather than modeling shared dynamics between two time-series (see section 2.1). Finally, prior multimodal learning algorithms do not explicitly tease apart the shared vs. residual (disjoint) dynamics in a predictor (primary) time-series, but instead model the collective dynamics of two modalities in the same latent states (Abbaspourazad et al., 2021; Kramer et al., 2022; Ahmadipour et al., 2023).  \n\nHere we fill these methodological gaps by deriving a novel covariance-based SSID learning algorithm that (1) is applicable to generalized-linear processes, and (2) is capable, with its two-staged learning approach, of identifying with priority the shared dynamics between two processes before modeling residual (predictor-only) dynamics. To illustrate the method, we focus on the specific case of modeling Poisson-distributed discrete time-series while dissociating their shared dynamics with Gaussian-distributed continuous observations, which is of particular interest in neuroscience. However, we emphasize that our method can be extended to other output distributions in the generalizedlinear model family (section 5). We show that our method successfully dissociated the shared dynamics between Poisson and Gaussian observations both in simulations and on a public non-human primate (NHP) dataset of discrete population spiking activity recorded during continuous arm movements (O’Doherty et al., 2017). Further, compared with existing Poisson SSID methods, our method more accurately decoded movements from Poisson spiking activity using lower-dimensional latent states. Lastly, we discuss limitations and propose potential solutions and future research directions.  \n\n\n\n\\section{2 BACKGROUND  }\n\n\nOur method provides the new capability to dynamically model Poisson observations, while prioritizing identification of dynamics shared with Gaussian observations. We first review the existing SSID method for modeling Poisson observations, which serves as our baseline, as well as standard covariance-based SSID, to help with the exposition of our method in section 3.  \n\n\n\n\\section{2.1 SSID FOR POISSON LINEAR DYNAMICAL SYSTEMS (PLDSID)  }\n\n\nA Poisson linear dynamical system (PLDS) model is defined as  \n\n$$\n\\left\\{\\begin{array}{l l l}{\\mathbf{x}_{k+1}}&{=}&{A\\mathbf{x}_{k}+\\mathbf{w}_{k}}\\\\ {\\mathbf{r}_{k}}&{=}&{C_{\\mathbf{r}}\\mathbf{x}_{k}+b}\\\\ {\\mathbf{y}_{k}\\mid\\mathbf{r}_{k}}&{\\sim}&{\\mathrm{Poisson}(\\exp(\\mathbf{r}_{k}))}\\end{array}\\right.\n$$  \n\nwhere $\\mathbf{x}_{k}\\,\\in\\,\\mathbb{R}^{n_{x}}$ is the latent state variable and $\\mathbf{y}_{k}\\,\\in\\,\\mathbb{R}^{n_{y}}$ corresponds to discrete (e.g., neural spiking) observations which, conditioned on the latent process $\\mathbf{r}_{k}$ , is Poisson-distributed with a rate equal to the exponential of $\\mathbf{r}_{k}$ (i.e., log-rate). Finally, $\\mathcal{N}(\\mathbf{w}_{k};\\mathbf{0},Q)$ is state noise and $^{b}$ is a constant baseline log-rate. The PLDS model is commonly used for modeling Poisson process events, such as neural spiking activity (Smith & Brown, 2003; Truccolo et al., 2005; Lawhern et al., 2010; Buesing et al., 2012; Macke et al., 2015). Buesing et al. (2012) developed an SSID algorithm, termed PLDSID, to learn the PLDS model parameters $\\Theta\\,=\\,(A,C_{\\mathbf{r}},b,Q)$ given training samples $\\pmb{y}_{k}$ and hyperparameter $n_{x}$ corresponding to the latent state dimensionality.  \n\nThere exist standard covariance-based SSID algorithms (section 2.2) that can learn the parameters of a latent dynamical system given a future-past Hankel matrix, $\\pmb{H}$ , constructed from the crosscovariances of the system’s linear observations as (Van Overschee & De Moor, 1996; Katayama, 2005)  \n\n$$\nH:=\\mathrm{Cov}(\\mathbf{r}_{f},\\mathbf{r}_{p})=\\left[\\begin{array}{c c c c}{\\mathbf{\\Lambda}_{\\mathbf{r}_{i}}}&{\\mathbf{\\Lambda}_{\\mathbf{r}_{i-1}}}&{\\cdot\\cdot\\cdot}&{\\mathbf{\\Lambda}_{\\mathbf{r}_{1}}}\\\\ {\\mathbf{\\Lambda}_{\\mathbf{r}_{i+1}}}&{\\mathbf{\\Lambda}_{\\mathbf{r}_{i}}}&{\\cdot\\cdot\\cdot}&{\\mathbf{\\Lambda}_{\\mathbf{r}_{2}}}\\\\ {\\vdots}&{\\vdots}&{\\cdot\\cdot}&{\\vdots}\\\\ {\\mathbf{\\Lambda}_{\\mathbf{r}_{2i-1}}}&{\\mathbf{\\Lambda}_{\\mathbf{r}_{2i-2}}}&{\\cdot\\cdot\\cdot}&{\\mathbf{\\Lambda}_{\\mathbf{r}_{i}}}\\end{array}\\right],\\;\\;\\mathbf{r}_{f}:=\\left[\\begin{array}{c}{\\mathbf{r}_{i}}\\\\ {\\vdots}\\\\ {\\mathbf{r}_{2i-1}}\\end{array}\\right],\\mathbf{r}_{p}:=\\left[\\begin{array}{c}{\\mathbf{r}_{0}}\\\\ {\\vdots}\\\\ {\\mathbf{r}_{i-1}}\\end{array}\\right],\n$$  \n\nwhere the integer $i$ denotes the user-specified maximum temporal lag (i.e., horizon) used to construct $\\pmb{H}$ and $\\mathbf{\\Lambda}_{\\mathbf{r}_{\\tau}}\\,:=\\,\\mathrm{Cov}(\\mathbf{r}_{k+\\tau},\\mathbf{r}_{k})$ is the $\\tau$ -th lag cross-covariance for any timepoint $k$ , under time-stationary assumptions. Such covariance-based SSID algorithms, however, are not directly applicable to Poisson-distributed observations (section 2.3). This is because the log-rates $\\mathbf{r}_{k}$ that are linearly related to the latent states in equation (1) are not observable in practice – rather, only a stochastic Poisson emission from them (i.e., $\\mathbf{y}_{k}^{\\sf},$ ) is observed. As a result, the second moments constituting $\\pmb{H}$ (i.e., $\\Lambda_{\\mathbf{r}_{\\tau}}$ ) cannot be directly estimated. The critical insight by Buesing et al. (2012) was to leverage the log link function (i.e., $\\mathrm{{exp^{-1}}}$ ) and the known conditional distribution $\\mathbf{y}_{k}\\vert\\mathbf{r}_{k}$ to compute the first $\\left(\\pmb{\\mu}_{\\mathbf{r}^{\\pm}}\\right)$ and second $\\left(\\mathbf{A}_{\\mathbf{r}^{\\pm}}\\right)$ moments of the log-rate $\\mathbf{r}_{k}$ from the first $(\\pmb{\\mu}_{\\mathbf{y}^{\\pm}})$ and second $(\\mathbf{A}_{\\mathbf{y}^{\\pm}})$ moments of the discrete observations $\\mathbf{y}_{k}$ . The $\\pm$ denotes that moments are computed for the future-past stacked vector of observations $\\mathbf{r}^{\\pm}:=\\left[\\mathbf{r}_{f}^{T}\\right.\\quad\\mathbf{r}_{p}^{T}\\right]^{T}$ and $\\mathbf{y}^{\\pm}:=\\left[\\mathbf{y}_{f}^{T}\\quad\\mathbf{y}_{p}^{T}\\right]^{T}$ , where  \n\n$$\n\\mu_{\\mathbf{r}^{\\pm}}:=E[\\mathbf{r}^{\\pm}]\\quad\\mu_{\\mathbf{y}^{\\pm}}:=E[\\mathbf{y}^{\\pm}]\\quad\\Lambda_{\\mathbf{r}^{\\pm}}:=\\operatorname{Cov}(\\mathbf{r}^{\\pm},\\mathbf{r}^{\\pm})\\quad\\Lambda_{\\mathbf{y}^{\\pm}}:=\\operatorname{Cov}(\\mathbf{y}^{\\pm},\\mathbf{y}^{\\pm}).\n$$  \n\nTo compute moments of the log-rate, Buesing et al. (2012) derived the following moment conversion  \n\n$$\n\\begin{array}{r c l}{\\mu_{\\mathbf{r}_{m}^{\\pm}}}&{=}&{2\\ln(\\pmb{\\mu}_{\\mathbf{y}_{m}^{\\pm}})-\\frac{1}{2}\\ln(\\Lambda_{\\mathbf{y}_{m_{m}}^{\\pm}}+\\pmb{\\mu}_{\\mathbf{y}_{m}^{\\pm}}^{2}-\\pmb{\\mu}_{\\mathbf{y}_{m}^{\\pm}})}\\\\ {\\Lambda_{\\mathbf{r}_{m m}^{\\pm}}}&{=}&{\\ln(\\Lambda_{\\mathbf{y}_{m m}^{\\pm}}+\\pmb{\\mu}_{\\mathbf{y}_{m}^{\\pm}}^{2}-\\pmb{\\mu}_{\\mathbf{y}_{m}^{\\pm}})-\\ln(\\pmb{\\mu}_{\\mathbf{y}_{m}^{\\pm}}^{2})}\\\\ {\\Lambda_{\\mathbf{r}_{m n}^{\\pm}}}&{=}&{\\ln(\\Lambda_{\\mathbf{y}_{m n}^{\\pm}}+\\pmb{\\mu}_{\\mathbf{y}_{m}^{\\pm}}^{2}\\mu_{\\mathbf{y}_{n}^{\\pm}})-\\ln(\\pmb{\\mu}_{\\mathbf{y}_{m}^{\\pm}}\\mu_{\\mathbf{y}_{n}^{\\pm}})}\\end{array}\n$$  \n\nwhere $m\\neq n$ correspond to different indices of the first and second moments of the future-past stacked observation vectors $\\mathbf{r}^{\\pm}$ and $\\mathbf{y}^{\\pm}$ , and $n,m=1,\\cdots,K n_{y}$ where $K$ is the total number of time points. With the first and second moments computed in the moment conversion above, the baseline log rate $^{b}$ parameter is read off the first $n_{y}$ rows of $\\pmb{\\mu}_{\\mathbf{r}^{\\pm}}$ and the Hankel matrix, $\\pmb{H}$ , is constructed as per equation (2). From here, it is possible to proceed with the standard covariancebased SSID algorithm for Gauss-Markov models using $\\pmb{H}$ , as outlined next.  \n\n\n\n\\section{2.2 STANDARD COVARIANCE-BASED SSID  }\n\n\nGiven an $\\pmb{H}$ matrix, covariance-based SSID first decomposes $\\pmb{H}$ into a product of observability $(\\mathbf{\\boldsymbol{\\Gamma}_{r}})$ and controllability $(\\Delta)$ matrices as (Van Overschee & De Moor, 1996; Katayama, 2005)  \n\n$$\nH^{\\mathrm{SUD}}\\Gamma_{\\mathbf{r}}\\Delta=\\left[\\begin{array}{c}{C_{\\mathbf{r}}}\\\\ {C_{\\mathbf{r}}A}\\\\ {\\vdots}\\\\ {C_{\\mathbf{r}}\\dot{A}^{i-1}}\\end{array}\\right]\\left[A^{i-1}G\\quad\\cdot\\cdot\\quad A G\\quad G\\right]\n$$  \n\nwhere $G:=\\operatorname{Cov}(\\mathbf{x}_{k+1},\\mathbf{r}_{k})$ . The factorization of $\\pmb{H}$ is done by computing a SVD of $\\pmb{H}$ and keeping the top $n_{x}$ singular values and corresponding singular vectors. Note that the rank of $\\pmb{H}$ must be at least $n_{x}$ in order to identify a model with a latent dimension of $n_{x}$ . Thus, the user-specified horizon $i$ must satisfy $i\\times n_{y}\\ge n_{x}$ . From the factors of $\\pmb{H}$ , $C_{\\mathbf{r}}$ is read off as the first $n_{y}$ rows of $\\mathbf{\\Gamma_{y}}$ and $\\pmb{A}$ is learned by solving $\\overline{{\\mathbf{T}}}_{\\mathbf{r}}=\\underline{{\\mathbf{T}}}_{\\mathbf{r}}A$ , where $\\overline{{\\Gamma}}_{\\mathrm{r}}$ and $\\underline{{\\mathbf{\\delta}}}_{\\mathbf{r}}$ denote $\\Gamma_{\\mathrm{r}}$ from which the top or bottom $n_{y}$ rows have been removed, respectively. This optimization problem has the following closed-form leastsquares solution $A=\\underline{{\\Gamma}}_{\\mathbf{r}}^{\\dagger}\\overline{{\\Gamma}}_{\\mathbf{r}}$ , with $\\dagger$ denoting the pseudo-inverse operation. Discussion regarding learning the state noise covariance model parameter $Q$ is postponed to section 3.2.3 below.  \n\n\n\n\\section{2.3 CHALLENGES OF DEVELOPING COVARIANCE- VS PROJECTION-BASED SSID METHODS  }\n\n\nAt a high-level, there exist two common approaches for subspace identification (Van Overschee & De Moor, 1996; Katayama, 2005): (i) covariance-based methods (e.g., Buesing et al. (2012); Ahmadipour et al. (2023)) that aim to learn all model parameters based on the second-order statistics of the observations and not the observation time-series directly, and (ii) projection-based methods (e.g., Sani et al. (2021); Vahidi et al. (2023)) that make direct use of the observation time-series via linear projections. Projection-based methods are often used to model Gaussian time-series but are not applicable to Poisson observations, which instead require a covariance-based approach since the latent log-firing rates (r in equation (1)) are unobserved. To achieve our aim (i.e., modeling a generalized-linear process with prioritized identification of shared dynamics), we need to develop a novel covariance-based subspace identification algorithm, which presents the following challenges:  \n\n1. Covariance-based methods, including PLDSID (Buesing et al., 2012), do not guarantee a valid set of parameters that satisfy the positive semidefinite covariance sequence requirements (Van Overschee & De Moor, 1996). To address this challenge, we use the optimization approach outlined in section 3.2.3 to ensure validity of noise statistics and enable inference from the learned model.  \n\n2. We could not rely on time-series projections to isolate the residual predictor process dynamics at the beginning of the second stage (Sani et al., 2021). As a result, we derived a new least-squares problem for learning the components of the state transition matrix $\\pmb{A}$ corresponding to the unique predictor process dynamics (section 3.2.2), without changing the shared components learned in the first stage (section 3.2.1). By doing so, prioritized learning of shared dynamics is preserved.  \n\n\n\n\\section{3 METHOD  }\n\n\n\n\n\\section{3.1 MODELING SHARED DYNAMICS BETWEEN POISSON AND GAUSSIAN OBSERVATIONS  }\n\n\nThe PLDS model (equation (1)) is for modeling Poisson observations on their own rather than with Gaussian observations. To enable this capability, we write the following Poisson-Gaussian linear dynamical system model  \n\n$$\n\\left\\{\\begin{array}{l l l}{\\mathbf{x}_{k+1}}&{=}&{A\\mathbf{x}_{k}+\\mathbf{w}_{k}}\\\\ {\\mathbf{z}_{k}}&{=}&{C_{\\mathbf{z}}\\mathbf{x}_{k}+\\boldsymbol{\\epsilon}_{k}}\\\\ {\\mathbf{r}_{k}}&{=}&{C_{\\mathbf{r}}\\mathbf{x}_{k}+\\boldsymbol{b}}\\\\ {\\mathbf{y}_{k}\\mid\\mathbf{r}_{k}}&{\\sim}&{\\mathrm{Poisson}(\\exp(\\mathbf{r}_{k}))}\\end{array}\\right.\n$$  \n\nwhere $\\mathbf{z}_{k}\\ \\in\\ \\mathbb{R}^{n_{z}}$ represents continuous observations (e.g., arm movements), $\\epsilon_{k}$ represents their noise (either white, i.e., zero-mean temporally uncorrelated Gaussian noise, or colored, i.e., zeromean temporally correlated Gaussian noise), and $\\mathbf{y}_{k}\\in\\mathbb{R}^{n_{y}}$ represents the discrete observations (i.e., neural spiking). Further, we introduce a block structure to the system (Sani et al., 2021) that allows us to dissociate shared latents from those that drive the Poisson observations only. Specifically,  \n\n$$\nA=\\left[A_{11}\\quad\\begin{array}{c}{\\mathbf{0}}\\\\ {A_{22}}\\end{array}\\right]\\quad C_{\\mathbf{z}}=\\left[C_{\\mathbf{z}}^{(1)}\\quad\\mathbf{0}\\right]\\quad C_{\\mathbf{r}}=\\left[C_{\\mathbf{r}}^{(1)}\\quad C_{\\mathbf{r}}^{(2)}\\right]\\quad\\mathbf{x}=\\left[\\mathbf{x}^{(1)}\\right]\n$$  \n\nwhere xk $\\mathbf{x}_{k}^{(1)}\\,\\in\\,\\mathbb{R}^{n_{1}}$ corresponds to latent states that drive both $\\mathbf{z}_{k}$ and $\\mathbf{y}_{k}$ , and $\\mathbf{x}_{k}^{(2)}\\,\\in\\,\\mathbb{R}^{n_{x}-n_{1}}$ to states that only drive $\\mathbf{y}_{k}$ . The parameter $\\pmb{G}$ can also be written in block partition format such that  \n\n$$\nG=E\\left[\\left[\\mathbf{x}_{k+1}^{\\left(1\\right)}\\right]\\mathbf{r}_{k}^{T}\\right]-E\\left[\\left[\\mathbf{x}_{k+1}^{\\left(1\\right)}\\right]\\right]E[\\mathbf{r}_{k}]^{T}=\\left[E[\\mathbf{x}_{k+1}^{\\left(1\\right)}\\mathbf{r}_{k}^{T}]\\right]-\\left[E[\\mathbf{x}_{k+1}^{\\left(1\\right)}]E[\\mathbf{r}_{k}]^{T}\\right]=\\left[\\mathbf{G}^{\\left(1\\right)}\\right].\n$$  \n\nOur method, termed PG-LDS-ID (Poisson-Gaussian linear dynamical system identification), learns the model parameters, i.e., $\\boldsymbol{\\Theta}^{'}=(A,C_{\\mathbf{z}},C_{\\mathbf{r}},b,Q)$ , given training samples $\\pmb{y}_{k}$ and $z_{k}$ and hyperparameters $n_{1}$ and $n_{2}=n_{x}-n_{1}$ denoting the shared and residual latent dimensionalities.3).  \n\n\n\n\\section{3.2 PG-LDS-ID  }\n\n\nPG-LDS-ID uses a two-staged learning approach to model Poisson time-series while prioritizing identification of the dynamics shared with Gaussian observations. During stage 1, shared dynamics are learned using both observations. In stage 2, any residual dynamics in the predictor observations are optionally learned.1). Note, predictor refers to the data stream whose modeling is of primary interest (the Poisson observations here) and that is used to predict the secondary data stream (the Gaussian observations). For example, Poisson observations are the predictor within the context of decoding continuous behaviors from discrete population spiking activity. The roles can be swapped without loss of generality and the designation is made clear in equation (8) below.  \n\nIn the first stage, our algorithm identifies the parameter set corresponding to the shared dynamical subspace, $(A_{11},C_{\\mathbf{r}}^{(1)},C_{\\mathbf{z}},b)$ , given hyperparameter $n_{1}$ and using both Gaussian and Poisson observations, $\\mathbf{z}_{k}$ and $\\mathbf{y}_{k}$ .1.4)  \n\n$$\n\\begin{array}{r l r}{{\\Lambda_{\\mathbf{z}_{f_{m}}\\mathbf{r}_{p_{n}}}}}&{{}\\!=}&{\\operatorname{Cov}({\\mathbf{z}_{f_{m}},\\mathbf{y}_{p_{n}}})\\,/\\,\\mu_{{\\mathbf{y}_{p_{n}}}}.}\\end{array}\n$$  \n\nNext, we use these moments to construct a Hankel matrix between future continuous observations and past log-rates of the discrete observations  \n\n$$\nH_{\\mathbf{zr}}:=\\operatorname{Cov}(\\mathbf{z}_{f},\\mathbf{r}_{p})={\\left[\\begin{array}{l l l l}{\\mathbf{\\Lambda}_{\\mathbf{A}\\mathbf{zr}_{i}}}&{\\mathbf{\\Lambda}_{\\mathbf{A}\\mathbf{zr}_{i-1}}}&{\\cdot\\cdot\\cdot}&{\\mathbf{\\Lambda}_{\\mathbf{A}\\mathbf{zr}_{1}}}\\\\ {\\mathbf{\\Lambda}_{\\mathbf{zr}_{i+1}}}&{\\mathbf{\\Lambda}_{\\mathbf{A}\\mathbf{zr}_{i}}}&{\\cdot\\cdot\\cdot}&{\\mathbf{\\Lambda}_{\\mathbf{A}\\mathbf{zr}_{2}}}\\\\ {\\vdots}&{\\vdots}&{\\cdot\\cdot}&{\\vdots}\\\\ {\\mathbf{\\Lambda}_{\\mathbf{zr}_{2i-1}}}&{\\mathbf{\\Lambda}_{\\mathbf{A}\\mathbf{zr}_{2i-2}}}&{\\cdot\\cdot\\cdot}&{\\mathbf{\\Lambda}_{\\mathbf{A}\\mathbf{zr}_{i}}}\\end{array}\\right]}\\,,\\quad\\mathbf{z}_{f}:={\\left[\\begin{array}{l}{\\mathbf{z}_{i}}\\\\ {\\vdots}\\\\ {\\mathbf{z}_{2i-1}}\\end{array}\\right]}\n$$  \n\nwith $\\mathbf{r}_{p}$ defined as in equation (2). Although equation (8) uses the same horizon for both observations, in practice we implement the method for a more general version with distinct horizon values $i_{\\mathbf{y}}$ for the discrete observations and $i_{\\mathbf{z}}$ for the continuous observations, resulting in $H_{\\mathbf{z}\\mathbf{r}}\\in\\mathbb{R}^{i_{\\mathbf{z}}\\ast\\bar{n_{z}}\\times i_{\\mathbf{y}}\\ast n_{y}}$ . This allows users to independently specify the horizons for the two observations, which can improve modeling accuracy especially if the two observations have very different dimensionalities (see section 4.3.1). Further, and importantly, by using $\\mathbf{z}$ as the future observations in the Hankel matrix, we learn a dynamical model wherein Poisson observations $\\mathbf{y}$ can be used to predict the Gaussian observations $\\mathbf{z}$ . After constructing $H_{\\mathrm{zr}}$ , we decompose it using SVD and keep the top $n_{1}$ singular values and their corresponding singular vectors  \n\n$$\nH_{\\mathbf{zr}}\\frac{\\mathrm{sy}0}{=}\\Gamma_{\\mathbf{z}}\\boldsymbol{\\Delta}^{(1)}=\\left[\\begin{array}{c}{C_{\\mathbf{z}}}\\\\ {C_{\\mathbf{z}}A_{11}}\\\\ {\\vdots}\\\\ {C_{\\mathbf{z}}A_{11}^{i-1}}\\end{array}\\right]\\left[A_{11}^{i-1}G^{(1)}\\quad\\cdot\\cdot\\cdot\\quad A_{11}G^{(1)}\\quad G^{(1)}\\right]\n$$  \n\n At this point, we extract $C_{\\mathbf{z}}$ by reading off the first $n_{z}$ rows of $\\mathbf{\\Gamma}_{\\mathbf{T}_{\\mathbf{z}}}$ . To extract $C_{r}^{(1)}$ we first form $\\pmb{H}$ per equation (2) and extract the observability matrix for $\\mathbf{r}$ associated with the shared latent dynamics, $\\mathbf{T}_{\\mathbf{y}}^{(\\mathrm{i})}$ , by right multiplying $\\pmb{H}$ with the pseudoinverse of $\\Delta^{(1)}$  \n\n$$\nH\\Delta^{(1)\\dagger}=\\Gamma_{\\bf r}^{(1)}=\\left[\\begin{array}{c}{C_{\\bf r}^{(1)}}\\\\ {C_{\\bf r}^{(1)}A_{11}}\\\\ {\\vdots}\\\\ {C_{\\bf r}^{(1)}A_{11}^{i-1}}\\end{array}\\right]\\,\\,.\n$$  \n\n The baseline log rate $^{b}$ is read off the first $n_{y}$ rows of $\\pmb{\\mu}_{\\mathbf{r}^{\\pm}}$ computed in the moment conversion from equation (3). Lastly, to learn the shared dynamics summarized by the parameter $A_{11}$ , we solve the optimization problem $\\underline{{\\Delta}}^{(1)}\\;=\\;{\\cal A}_{11}\\overline{{\\Delta}}^{(\\dot{1})}$ where $\\underline{{\\Delta}}^{(1)}$ and $\\overline{{\\Delta}}^{(\\mathrm{i})}$ denote $\\Delta^{(1)}$ from which $n_{y}$ columns have been removed from the right or left, respectively. The closed-form least-squares solution for this problem is $A_{11}=\\underline{{\\Delta}}^{(1)}(\\overline{{\\Delta}}^{(1)})^{\\dagger}$ . This concludes the learning of the desired parameters $(A_{11},C_{\\mathbf{r}}^{(1)},C_{\\mathbf{z}},b)$ , given hyperparameter $n_{1}$ , in stage 1.  \n\n\n\n\\section{3.2.2 STAGE 2: RESIDUAL DYNAMICS  }\n\n\nAfter learning the shared dynamics, our algorithm can learn the residual dynamics in the predictor observations that were not captured by $\\mathbf{x}_{k}^{(1)}$ . Specifically, we learn the remaining parameters from  \n\nequation (6): $\\begin{array}{r l}{\\left(\\left[A_{21}\\right.}&{{}A_{22}\\right],C_{\\mathbf{r}}^{(2)}\\right)}\\end{array}$ , with hyperparameter $n_{2}=n_{x}-n_{1}$ determining the unshared latent dimensionality. To do so, we first compute a “residual” Hankel matrix, $H^{(2)}$ , using $\\mathbf{\\boldsymbol{\\Gamma}}_{\\mathbf{r}}^{(1)}$ and $\\Delta^{(1)}$ from stage 1 and decompose it using SVD, keeping the first $n_{2}$ singular values and vectors  \n\n$$\n\\begin{array}{r}{\\pmb{H}^{(2)}=\\pmb{H}-\\Gamma_{\\mathbf{r}}^{(1)}\\pmb{\\Delta}^{(1)}\\overset{\\mathrm{SVD}}{=}\\pmb{\\Gamma}_{\\mathbf{r}}^{(2)}\\pmb{\\Delta}^{(2)}.}\\end{array}\n$$  \n\nWith $C_{\\mathbf{r}}^{(2)}$ , which corresponds to the first $n_{y}$ rows of $\\mathbf{\\boldsymbol{\\Gamma}}_{\\mathbf{r}}^{(2)}$ , we construct $C_{r}\\ =\\ \\left[C_{r}^{(1)}\\quad C_{r}^{(2)}\\right]$ .1): $\\begin{array}{r l}{\\pmb{\\Delta}=}&{{}\\left[\\pmb{A}^{i-1}\\pmb{G}\\ \\ \\cdot\\cdot\\cdot\\ \\ \\ A\\pmb{G}\\ \\ \\ G\\right]=\\left[\\pmb{\\Delta}^{(1)}\\right].}\\end{array}$ = ∆∆(2) . Given ∆, we next extract $\\left[A_{21}\\quad A_{22}\\right]$ by solving the problem $\\underline{{\\Delta}}^{(2)}=\\left[\\mathbf{A}_{21}\\quad\\mathbf{A}_{22}\\right]\\overline{{\\Delta}}$ where  \n\n$$\n\\underline{{\\Delta}}^{(2)}:=\\left[\\left[A_{21}\\quad A_{22}\\right]A^{i-2}G\\quad\\cdot\\cdot\\quad\\left[A_{21}\\quad A_{22}\\right]G\\right],\\quad\\overline{{\\Delta}}:=\\left[A^{i-2}G\\quad\\cdot\\cdot\\quad G\\right].\n$$  \n\nConcatenating all the sub-blocks together $\\begin{array}{r}{\\pmb{A}=\\left[\\pmb{A}_{11}\\pmb{\\omega}_{1}^{\\textbf{\\textsf{0}}}\\right],}\\\\ {\\pmb{A}_{21}\\pmb{\\omega}_{22}\\bigg],}\\end{array}$ , we now have all model parameters, $(A,C_{\\mathbf{r}},C_{\\mathbf{z}},b)$ , given hyperparameters $n_{1}$ and $n_{2}$ , except state noise covariance $Q$ .  \n\n\n\n\\section{3.2.3 NOISE STATISTICS  }\n\n\nStandard SSID algorithms (e.g., section 2.2) learn linear SSMs of the following form  \n\n$$\n\\left\\{\\begin{array}{l c l}{\\mathbf{x}_{k+1}}&{=}&{A\\mathbf{x}_{k}+\\mathbf{w}_{k}}\\\\ {\\mathbf{r}_{k}}&{=}&{C_{\\mathbf{r}}\\mathbf{x}_{k}+\\mathbf{v}_{k}}\\end{array}\\right.\n$$  \n\nwhere the new term $\\mathcal{N}(\\mathbf{v}_{k};\\mathbf{0},R)$ corresponds to observation noise. State noise, $\\mathbf{w}_{k}$ , and observation noise, $\\mathbf{v}_{k}$ , can have a non-zero instantaneous cross-covariance $S=\\operatorname{Cov}\\!\\left(\\mathbf{w}_{k},\\mathbf{v}_{k}\\right)$ . SSID in general does not assume any restrictions on the noise statistics. However, the Poisson observation model (equations (1) and (5)) has no additive Gaussian noise for $\\mathbf{r}_{k}$ and instead exhibits Poisson noise in $\\mathbf{y}_{k}$ , when conditioned on $\\mathbf{r}_{k}$ . This means that $\\mathbf{v}_{k}\\,=\\,{\\bf0}$ in equation (5), and thus $\\scriptstyle R\\;=\\;{\\bf0}$ and $S\\,=\\,{\\bf0}$ . Imposing these constraints is important for accurate parameter identification for Poisson observations, but was not previously addressed by Buesing et al. (2012). Thus, we require our algorithm to find a complete parameter set $\\Theta^{\\prime}$ that is close to the learned $(A,C_{\\mathbf{r}},C_{\\mathbf{z}},b)$ from the two stages in sections 3.2.1 and 3.2.2 and imposes the noise statistic constraints $\\scriptstyle R\\;=\\;{\\bf0}$ and $S=\\mathbf{0}$ . To do this, inspired by Ahmadipour et al. (2023), we form and solve the following convex optimization problem to satisfy the noise statistics requirements  \n\n$$\n\\begin{array}{r}{\\operatornamewithlimits{m i n i m i z e}\\quad\\|S(\\Lambda_{\\mathbf{x}})\\|_{F}^{2}+\\|R(\\Lambda_{\\mathbf{x}})\\|_{F}^{2}\\quad\\mathrm{such~that~}\\Lambda_{\\mathbf{x}}\\succeq0,\\ Q(\\Lambda_{\\mathbf{x}})\\succeq0,\\ R(\\Lambda_{\\mathbf{x}})\\succeq0}\\end{array}\n$$  \n\nwhere $\\mathbf{\\calN}_{\\mathbf{x}}\\,:=\\,\\mathrm{Cov}(\\mathbf{x}_{k},\\mathbf{x}_{k})$ denotes the latent state covariance and the following covariance relationships, derived from equation (11) (Van Overschee & De Moor, 1996), hold  \n\n$$\n\\left\\{\\begin{array}{l l l l l}{Q(\\Lambda_{\\mathbf{x}})}&{=}&{\\Lambda_{\\mathbf{x}}}&{-}&{A\\Lambda_{\\mathbf{x}}A^{T}}\\\\ {R(\\Lambda_{\\mathbf{x}})}&{=}&{\\Lambda_{\\mathbf{r}_{0}}}&{-}&{C_{\\mathbf{r}}\\Lambda_{\\mathbf{x}}C_{\\mathbf{r}}^{T}}\\\\ {S(\\Lambda_{\\mathbf{x}})}&{=}&{G}&{-}&{A\\Lambda_{\\mathbf{x}}C_{\\mathbf{r}}^{T}.}\\end{array}\\right.\n$$  \n\nThis approach has multiple benefits. First, it finds noise statistics that are consistent with the assumptions of the model (e.g., $\\scriptstyle R\\;=\\;{\\bf0}$ ). Second, it enforces the validity of learned parameters, i.e., parameters corresponding to a valid positive semidefinite covariance sequence (see section 4.3).4). Combining the previously found parameters and the matrix $Q$ that corresponds to the minimizing solution $\\mathbf{\\DeltaA_{x}}$ of equation (12), we have the full parameter set $\\Theta^{\\prime}=(A,C_{\\mathbf{r}},C_{\\mathbf{z}},b,Q)$ . We used Python’s CVXPY package to solve the semidefinite programming problem defined in equation (12) (Diamond & Boyd, 2016; Agrawal et al., 2018). For all of our comparisons against baseline, we learned the noise statistics associated with PLDSID’s identified parameters using this approach, keeping the rest of the algorithm the same.  \n\n\n\n\\section{4 EXPERIMENTAL RESULTS  }\n\n\n\n\n\\section{4.1 SHARED DYNAMICS ARE ACCURATELY IDENTIFIED IN SIMULATIONS  }\n\n\nWe simulated Poisson and Gaussian observations from random models as per equation (5) to evaluate how well our method identified the shared dynamics between the two observations.5.1). We computed two performance metrics: 1) the normalized eigenvalue error between ground truth and identified shared dynamical modes (i.e, the eigenvalues of $A_{11}$ in equation (6)), and 2) the predictive power of the model when using discrete Poisson observations to predict continuous Gaussian observations in a held-out test set. This second metric allowed us to test our hypothesis that PG-LDS-ID’s explicit modeling of the shared subspace improved decoding of Gaussian observations from Poisson observations compared with PLDSID (Buesing et al., 2012). To compute the first metric for PLDSID, which does not explicitly model shared dynamics, we needed to select the $n_{1}$ modes identified from the Poisson time-series only that were the most representative of the Gaussian time-series. To do so, we first trained PLDSID on Poisson observations and extracted the latent states.4). We computed the eigenvalues associated with the top $n_{1}$ most predictive latent states, which we considered as the shared modes identified by PLDSID. We computed the normalized eigenvalue error as $\\left|\\Psi_{\\mathrm{true}}-\\Psi_{\\mathrm{id}}\\right|_{F}/\\left|\\Psi_{\\mathrm{true}}\\right|_{F}$ , where $\\Psi_{\\mathrm{true}}$ and $\\Psi_{\\mathrm{id}}$ denote vectors containing the true and learned shared eigenvalues and $|\\cdot|_{F}$ denotes the Frobenius norm.  \n\n![](images/bbb6dd9a1a2a20020dde8ecc08330c34bd95f5aa86e072997ba9a08e25286c2a.jpg)  \nFigure 1: In simulations, PG-LDS-ID more accurately learns the shared dynamical modes and better predicts the Gaussian observations from Poisson observations, especially in lowdimensional regimes. Solid traces show the mean and the shaded areas denote the standard error of the mean (s.e.m.) for each condition. (a-b) Results for random models. Both the prediction correlation coefficient for the Gaussian observations in (a) and the normalized identification error of the shared dynamical modes (in log10 scale) in (b) are shown as a function of training samples used to learn the model parameters. (c-d) Same as (a-b) but for models with fixed shared $(n_{1}=4)$ ) and residual $n_{2}=12\\$ ) latent dimensions in the Poisson observations. PG-LDS-ID stage 1 used a dimensionality given by $\\operatorname*{min}(4,n_{x})$ . For configurations wherein learned $n_{x}$ is smaller than true $n_{1}$ , we substituted missing modes with 0 prior to computing the normalized error.  \n\nIn our first simulation experiment, we generated 50 random systems and studied the effect of training set size on learning. We used 1e2, 1e3, 1e4, 1e5 or 1e6 samples to train models and tested them on 1e6 samples of independent held-out data (figure 1). We found that our method required substantially fewer training samples $_{\\sim1\\mathrm{e}4}$ samples compared to PLDSID’s ${\\sim}1\\mathrm{e}5\\$ ) to reach ideal (i.e., ground truth) prediction (figure 1a). Similarly, our method more accurately identified the shared dynamical modes compared to PLDSID even when methods had increasingly more training samples (figure 1b). In our second simulation experiment, we studied the effect of latent state dimension on learning. We generated 50 systems with fixed dimensions for shared and total latent states given by $n_{1}=4$ and $n_{x}=16$ , respectively. We swept the learned latent state dimension from 1 to the true dimensionality of $n_{x}=16$ , with the dimensionality of shared dynamics set to min(current $n_{x},n_{1})$ . We found that our method identified the correct shared modes with very small errors using only 4 latent state dimensions; in contrast, PLDSID did not reach such low error rates even when using higher latent state dimensions (figure 1d). In terms of predictive power, our method achieved close-to peak performance even when using as few as 4 latent states whereas PLDSID required much larger latent states dimensions, of around 16, to do so (figure 1c). Taken together, these results show the power of PG-LDS-ID for performing dimensionality reduction on Poisson observations while prioritizing identification of shared dynamics with a secondary Gaussian data stream.  \n\n4.2 MODELING SHARED AND RESIDUAL DYNAMICS IN POISSON POPULATION NEURAL SPIKING ACTIVITY IMPROVES MOTOR DECODING  \n\nAs a demonstration on real data, we used our algorithm to model the shared dynamics between discrete population neural spiking activity and continuous arm movements in a publicly available NHP dataset from the Sabes lab (O’Doherty et al., 2017). The dataset is of a NHP moving a 2Dcursor in a virtual reality environment based on fingertip position. We use the 2D cursor position and velocity as the continuous observations z. We removed channels that had average firing rates less than $0.5\\:\\mathrm{Hz}$ or greater than $100\\;\\mathrm{Hz}$ . Similar to Lawlor et al. (2018), we also removed channels that were correlated with other channels using a correlation coefficient threshold of 0.4. For all methods we used $50\\mathrm{ms}$ binned multi-unit spike counts for the discrete observations y. We evaluated decoding performance of learned models using five-fold cross validation across six recording sessions. We performed cross-validation using randomly-selected, non-overlapping subsets of 15 channels $(n_{y}=$ 15) within each session. We used a nested inner cross-validation to select hyperparameters per fold based on the prediction CC of kinematics in the training data. Hyperparameters in this context were discrete horizon $i_{\\mathbf{y}}$ , continuous horizon $i_{\\mathbf{z}}$ , and time lag, which specifies how much the neural time-series should be lagged to time-align with the corresponding behavioral time-series (Moran & Schwartz, 1999; Shoham et al., 2005; Pandarinath et al., 2018). We swept $i_{\\mathbf{y}}$ values of 5 and 10 time bins, $i_{\\mathbf{z}}$ values of 10, 20, 22, 25, 28, and 30 time bins; and lag values of 0, 2, 5, 8, and 10 time bins. To train PG-LDS-ID, we use the shared dynamics dimensionality of $n_{1}=\\operatorname*{min}(\\operatorname{current}n_{x},8)$ . We chose a maximum $n_{1}$ of 8 because behavior decoding roughly plateaued at this dimension.  \n\nCompared with PLDSID, our method learned models that led to better behavioral decoding at all latent state dimensions (figure 2a) and achieved a higher behavior decoding at the maximum latent state dimension. This result suggests that our method better learns the shared dynamics between Poisson spiking and continuous movement observations due to its ability to dissociate shared vs. residual latent states in Poisson observations. Interestingly, despite the focus on learning the shared latent states in the first stage, PG-LDS-ID was also able to extract the residual latent states in Poisson observations because of its second stage. This led to PG-LDS-SID performing similarly to PLDSID in terms of peak neural self-prediction AUC while outperforming PLDSID in terms of peak behavior decoding (figure 2c). Indeed, even with the inclusion of just two additional latent states to model residual Poisson dynamics $\\mathit{n}_{2}\\,=\\,2$ , $n_{x}\\,=\\,10^{\\circ}$ ), neural self-prediction was comparable to models learned by PLDSID (figure 2b). Taken together, our method was extensible to real data and helped boost decoding performance, especially in low-dimensional latent regimes, by better identifying shared dynamics between Poisson and Gaussian observations.8 we also include preliminary results comparing against PLDS models fit using EM on a subset of this dataset.  \n\n\n\n\\section{4.3 LIMITATIONS  }\n\n\nPG-LDS-ID, similar to other SSID methods, uses a time-invariant model which may not be suitable if the data exhibits non-stationarity, e.g., in chronic neural recordings. In such cases one would need to intermittently refit the model or develop adaptive extensions (Ahmadipour et al., 2021). Moreover, as with other covariance-based SSID methods, our method may be sensitive to the accuracy of the empirical estimates of the first- and second-order moments. However, with increasing number of samples these empirical estimates will approach true statistical values, thereby improving overall performance, as seen in figure 1a-b.6). These issues can arise due to errors in the empirical estimates of the covariances and because these methods do not explicitly impose stability constraints on model parameters. Future work may consider incorporating techniques from control theory, such as mode stabilization and covariance matching, to help mitigate these limitations (Maciejowski, 1995; Lindquist & Picci, 1996; Byrnes et al., 1998; Alkire & Vandenberghe, 2002). Finally, our modeling approach can only provide an approximation of nonlinear dynamics/systems within the class of generalized-linear models, which have been shown to well-approximate nonlinear data in many applications, including modeling of neural and behavioral data.  \n\n![](images/4514feaec4ced8778485ca6ca2f3e698970975be1507c008963c56a4466ac407.jpg)  \nFigure 2: In NHP data, PG-LDS-ID improves movement decoding from Poisson population spiking activity. (a) Solid traces show the average cross-validated kinematic prediction CC and the shaded areas denote the s.e.m. for Poisson models of different latent dimensions learned by PG-LDS-ID (green) and PLDSID (orange). (b) Same as (a) but visualizing one-step ahead neural self-prediction AUC. (c) The left bar plots visualize the kinematic prediction CC and the right bar plots visualize the neural self-prediction AUC for models of latent dimensionality $n_{x}\\,=\\,14$ . We used Wilcoxon signed-rank test to measure significance. Asterisks in kinematic prediction CC plot indicate statistical significance with $p<0.0005$ ; neural self-prediction AUCs were not significantly different at $n_{x}=14$ . (d) Example decoding of cursor $\\mathrm{(x,y)}$ position and velocity from test data.  \n\n\n\n\\section{5 DISCUSSION  }\n\n\nWe developed a novel analytical two-staged subspace identification algorithm termed PG-LDS-ID for modeling Poisson data streams while dissociating the dynamics shared with Gaussian data streams. Using simulations and real NHP data, we demonstrated that our method successfully achieves this new capability and thus, compared to existing Poisson SSID methods, more accurately identifies Poisson dynamics that are shared with Gaussian observations. Furthermore, this capability allows our method to improve decoding performance despite using lower-dimensional latent states and requiring a fewer number of training samples. Although we specifically focused on modeling Gaussian and Poisson observations, our algorithm can be extended to alternate distributions described with generalized-linear models. Our algorithm only requires the second-order moments after moment conversion (see equations (2), (3), (7), (8)). Because the moment conversion algorithm can be modified for the desired link function in generalized-linear models, as explained by Buesing et al. (2012), we can combine our method with the appropriate moment conversion to extend it to other non-Gaussian and non-Poisson observation distributions. Due to the high-prevalence of generalized-linear models across various application domains (e.g., biomedical engineering, neuroscience, finance, etc.), our method can be a general tool for modeling shared and residual dynamics of joint data streams with distinct observation distributions.  \n\n\n\n\\section{6 REPRODUCIBILITY STATEMENT  }\n\n\nWe have taken a few steps to ensure reproducibility of the results reported here. First, we are sharing the implementation of our algorithm, as supplementary material, along with example simulated data and a tutorial IPython notebook to demonstrate usage. Second, we used a publicly available dataset (O’Doherty et al., 2017) that can be easily accessed by anyone interested in reproducing the results reported in section 4.2. Finally, to further aid in reproducing results, we have also outlined the preprocessing and analyses steps we have taken in section 4.5.2.  \n\n\n\n\n"

    # Generate review using Best Mode
    reviews = reviewer.evaluate_best_mode(
        paper_context=paper_context,
        reviewer_num=4
    )

    print(reviews)



