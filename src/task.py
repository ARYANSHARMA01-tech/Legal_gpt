from crewai import Task
from textwrap import dedent

class LegalTasks():

    def classify_user_query_task(self, agent, user_input):
        return Task(
            description=dedent(f"""
                Analyze the user's input to classify the nature of the legal request.

                User input: {user_input}
                Your task is to:
                1. Determine the type of query: Legal Question, Case Summary, Form Help, Document Upload
                2. Identify if it requires retrieval, analysis, or drafting
                3. Route the query for further processing

                Clearly label the task type and required next step.
            """),
            expected_output="A task type label (e.g., 'Legal Question', 'Summary', 'Form Help') and routing recommendation.",
            agent=agent
        )

    def retrieve_legal_documents_task(self, agent, classified_query):
        return Task(
            description=dedent(f"""
                Retrieve relevant legal documents (laws, cases, acts) based on the user's classified query.

                Query: {classified_query}

                Your task is to:
                1. Use the query classification to search for relevant legal documents
                2. Return 2–3 high-relevance results
                3. Summarize the context of each retrieved document

                Highlight the importance of each result.
            """),
            expected_output="A list of 2–3 legal documents with summaries and justification for relevance.",
            agent=agent
        )

    def analyze_legal_documents_task(self, agent, retrieved_docs):
        return Task(
            description=dedent(f"""
                Analyze the legal documents and derive insights relevant to the user's query.

                Documents: {retrieved_docs}

                Your task is to:
                1. Interpret the key legal principles or rulings in the documents
                2. Relate them directly to the user's issue
                3. Provide a concise explanation suitable for a citizen or lawyer

                Maintain accuracy and clarity in legal interpretation.
            """),
            expected_output="A detailed analysis explaining how the documents apply to the user's case or query.",
            agent=agent
        )

    def draft_legal_response_task(self, agent, analysis_result):
        return Task(
            description=dedent(f"""
                Based on the analysis, draft a user-facing response or legal document.

                Input: {analysis_result}

                Your task is to:
                1. Create a clear, concise legal response tailored to the user type (citizen or lawyer)
                2. Maintain proper tone and legal accuracy
                3. Optionally draft a legal notice, summary, or form help

                Keep language simple unless the user is a legal expert.
            """),
            expected_output="A ready-to-send legal summary, answer, or draft document in text format.",
            agent=agent
        )

    def validate_legal_output_task(self, agent, drafted_output):
        return Task(
            description=dedent(f"""
                Review the final response or draft for factual correctness and hallucinations.

                Input: {drafted_output}

                Your task is to:
                1. Check the response for legal accuracy and factual grounding
                2. Flag any unsupported claims or hallucinated citations
                3. Confirm clarity and legality of the output

                Provide a validation status and optional improvement suggestions.
            """),
            expected_output="A pass/fail validation with comments and corrections if needed.",
            agent=agent
        )
