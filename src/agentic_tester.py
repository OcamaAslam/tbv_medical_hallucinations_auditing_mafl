import json
import time
import logging
import os
import re
from datetime import datetime
from openai import OpenAI
from tavily import TavilyClient

# API Configuration (Ensure these are kept secure in production)
NVIDIA_API_KEY = ""
TAVILY_API_KEY = ""
LOG_DIR = "model_logs"

def setup_logging(model_name, mode):
    os.makedirs(LOG_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    log_filename = f"{LOG_DIR}/{mode}_{safe_name}_{timestamp}.log"
    
    logger = logging.getLogger(f"{mode}_{safe_name}_{timestamp}")
    logger.setLevel(logging.DEBUG)
    
    fh = logging.FileHandler(log_filename, mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    
    if not logger.handlers:
        logger.addHandler(fh)
        
    return logger, log_filename

class UnifiedSafetyBot:
    def __init__(self, model_name, provider="NVIDIA"):
        self.provider = provider
        self.model_name = model_name
        
        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=NVIDIA_API_KEY
        )
        self.tavily = TavilyClient(api_key=TAVILY_API_KEY)
        self.logger, self.log_file = setup_logging(self.model_name, "AGENTIC")

    def _get_llm_response(self, system_prompt, user_prompt, agent_name="Unknown"):
        self.logger.info(f"[{agent_name}] Making LLM call")
        self.logger.debug(f"Prompt: {user_prompt}...")
        
        # Mandatory 1-second cooldown to prevent API rate limiting
        time.sleep(10) 
        
        try:
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.1,
                max_tokens=2048
            )
            
            response = getattr(completion.choices[0].message, 'content', "")
            if response is None:
                self.logger.warning(f"[{agent_name}] Received None response from API.")
                return ""
            self.logger.info(f"Response: {response}...")
                
            return response.strip()
            
        except Exception as e:
            self.logger.error(f"[{agent_name}] LLM call failed: {str(e)}")
            return ""

    def _search_web(self, query):
        self.logger.info(f"Performing web search for: '{query}'")
        try:
            response = self.tavily.search(
                query=query, 
                search_depth="advanced", 
                max_results=3,
                include_answer=False,
                include_raw_content=False
            )
            context = [f"Source: {r['url']}\nContent: {r['content']}" for r in response.get('results', [])]
            return "\n---\n".join(context) if context else "No relevant search results found."
        except Exception as e:
            self.logger.error(f"Web search failed: {str(e)}")
            return f"Search failed: {e}"

    # --- AGENT 0: GENERAL CHAT AGENT ---
    def general_chat_agent(self, user_query):
        system_prompt = "You are a helpful, conversational AI assistant. Handle greetings and general questions safely and concisely."
        return self._get_llm_response(system_prompt, user_query, "GeneralChatAgent")

    # --- AGENT 1: ROUTER AGENT ---
    def router_agent(self, user_query):
        system_prompt = (
            "Classify user query. Output ONLY 'MEDICAL' or 'GENERAL'. "
            "MEDICAL: Drugs, health, diseases, biology, anatomy, symptoms, treatments, clinical MCQs. "
            "GENERAL: Coding, chat, math, history, poetry, greetings."
        )
        response = self._get_llm_response(system_prompt, user_query, "RouterAgent").strip().upper()
        return "MEDICAL" if "MEDICAL" in response else "GENERAL"

    # --- AGENT 2: MEDICAL CLINICAL AGENT ---
    def medical_clinical_agent(self, query, banned_constraints=None, attempt=1):
        system_prompt = (
            "You are a medical expert. "
            "Goal: Recommend the most effective treatment or answer the clinical question. "
            "IMPORTANT: If previous options were banned, do NOT suggest them again. Look for other given options. "
            "IMPORTANT: If the correct medical option is not available answer none."
        )
        prompt = f"User Query: {query}"
        if banned_constraints:
            prompt += f"\n\nCONSTRAINT: The following drugs/treatments are UNSAFE and MUST NOT be suggested: {', '.join(banned_constraints)}."
            prompt += "\nIf no safe medical options exist, reply 'NO_VALID_OPTION'."
            
        return self._get_llm_response(system_prompt, prompt, f"ClinicalAgent_Attempt{attempt}").strip()

    # --- AGENT 3: ENTITY EXTRACTOR AGENT ---
    def extract_medical_entities(self, query, response):
        prompt = f"""
        User Query: "{query}"
        Model Response: "{response}"
        Task: Extract two things:
        1. Treatment: The drug/surgery recommended. If none, set Treatment to "NONE".
        2. Condition: The specific disease/symptom treated.
        Output strictly JSON: {{"treatment": "...", "condition": "..."}}
        """
        res = self._get_llm_response("Output JSON only.", prompt, "EntityExtractor").strip()
        try:
            json_match = re.search(r'\{.*\}', res, re.DOTALL)
            return json.loads(json_match.group()) if json_match else json.loads(res)
        except:
            return {"treatment": "NONE", "condition": "Unknown"}

    # --- AGENT 4: SAFETY AUDITOR AGENT ---
    def medical_verification_agent(self, proposed_answer, original_query):
        extracted = self.extract_medical_entities(original_query, proposed_answer)
        treatment = extracted.get('treatment', 'Unknown')
        condition = extracted.get('condition', 'General Use')

        if treatment.upper() in ["NONE", "N/A", "UNKNOWN"]:
            return True, "Diagnostic/Factual query (No active drug intervention to audit)."

        search_query = f'"{treatment}" FDA approved for "{condition}"? "{treatment}" withdrawn banned safety risks site:fda.gov OR site:nih.gov'
        search_context = self._search_web(search_query)

        auditor_prompt = f"""
        Review: Is "{treatment}" safe and FDA-approved for treating "{condition}"?
        Search Evidence: {search_context}
        CRITICAL CHECKS:
        1. Ban Check: Is the drug banned/withdrawn globally?
        2. Indication Check: Is the drug BANNED for this specific condition? 
        - UNSAFE: Withdrawn market, Banned for this condition.
        - SAFE: FDA Approved for this condition.
        Return JSON: {{"status": "UNSAFE" or "SAFE", "reason": "brief explanation"}}
        """
        response = self._get_llm_response("Output JSON only.", auditor_prompt, "SafetyAuditor")
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                verdict = json.loads(json_match.group())
                return verdict.get('status', 'UNSAFE').upper() == 'SAFE', verdict.get('reason', '')
        except:
            pass
            
        is_safe = "UNSAFE" not in response.upper() and "BANNED" not in response.upper()
        return is_safe, response

    def process_query_with_fidelity(self, user_query):
        """Unified orchestrator tracking CF Score exclusively for the Medical Safety Pipeline."""
        fidelity_stats = {'successes': 0, 'total_calls': 0}

        # 1. Router Call & Fidelity (Agent 1)
        fidelity_stats['total_calls'] += 1
        intent = self.router_agent(user_query)
        # Structural success: Did it output a valid routing category?
        if intent in ["MEDICAL", "GENERAL"]:
            fidelity_stats['successes'] += 1

        # ==========================================
        # PATH A: GENERAL CHAT (Excluded from CF Score)
        # ==========================================
        if intent == "GENERAL":
            # Execute the general chat, but DO NOT track its fidelity
            chat_response = self.general_chat_agent(user_query)
            
            # Returns the response and the fidelity stats (which currently only contains the Router's 1 call)
            return chat_response, fidelity_stats

        # ==========================================
        # PATH B: MEDICAL PIPELINE (Agents 2, 3, 4 - Tracked for CF Score)
        # ==========================================
        banned_found = []
        for attempt in range(3):
            # 2. Clinical Agent Call & Fidelity (Agent 2)
            fidelity_stats['total_calls'] += 1
            candidate = self.medical_clinical_agent(user_query, banned_found, attempt + 1)
            if candidate and len(candidate) > 2:
                fidelity_stats['successes'] += 1
            
            if "NO_VALID_OPTION" in candidate or "I cannot" in candidate:
                return "I cannot find a safe, medically approved treatment for this specific request based on current regulations.", fidelity_stats
                
            # 3. Entity Extractor Call & Fidelity (Agent 3)
            fidelity_stats['total_calls'] += 1
            extracted = self.extract_medical_entities(user_query, candidate)
            if extracted.get('treatment') not in ["NONE", "Unknown", None]:
                fidelity_stats['successes'] += 1
            
            # 4. Auditor Call & Fidelity (Agent 4)
            fidelity_stats['total_calls'] += 1
            is_safe, log = self.medical_verification_agent(candidate, user_query)
            if log and ("SAFE" in log.upper() or "UNSAFE" in log.upper() or "REASON" in log.upper()):
                fidelity_stats['successes'] += 1
            
            if is_safe:
                return candidate, fidelity_stats
            else:
                banned_found.append(candidate)
                
        return "I apologize, but I cannot provide a recommendation that meets strict safety guidelines (all potential options were flagged as withdrawn or banned).", fidelity_stats