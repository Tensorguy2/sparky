

Senior Data Scientist | AI & ML Engineer | Agentic Architect
 
EXECUTIVE SUMMARY
Lead AI Engineer and Data Scientist with over 8 years of industry experience developing production-ready ML, NLP, and Generative AI solutions across AWS, GCP, and Azure. Experienced in architecting multi-agent systems, RAG pipelines, forecasting models, and cloud-based AI services, with a strong emphasis on security, observability, scalability, and reliable deployment. Recent projects include multi-agent AI orchestration, LLM-powered decision-support systems, and enterprise AI platforms built for regulated, data-intensive environments. Expert Multi-Agent Architect, Ai Governance and Elavuation and Monitoring Systems.
 
TECHNICAL SKILLS


Programming & Scripting: Python, SQL, Java, Node.js, Shell Scripting, Bash, REST APIs, JSON, YAML
AI Engineering & LLM Application Development: Prompt Engineering, Structured Outputs, Function Calling, Tool Calling, JSON Schema Validation, LLM Workflows, Context Management, Token Optimization, Guardrail Design, Human-in-the-Loop Workflows
Agentic AI & Multi-Agent Orchestration: Agentic AI, Multi-Agent Systems, OpenAI SDK, OpenAI Agents SDK, LangChain, LangGraph, Google ADK, AutoGen, CrewAI, Agent Handoffs, Planning and Reasoning Workflows, Context Orchestration, Stateful Agent Workflows, Workflow Automation, Model Context Protocol (MCP), Agent-to-Agent Communication (A2A)
RAG, Search & Retrieval Systems: Retrieval-Augmented Generation (RAG), Context-Augmented Generation (CAG), Hybrid Search, Vector Search, Semantic Search, Keyword Search, Metadata Filtering, Document Chunking, Query Rewriting, Reranking, Citation Generation, Grounded Response Generation, Azure AI Search, Amazon OpenSearch Service, Pinecone
Embedding Models & Retrieval Optimization: OpenAI Embeddings, Cohere Embed v3, Voyage AI Embeddings, BGE Embeddings, E5 Embeddings, Embedding Evaluation, Similarity Search, Approximate Nearest Neighbor Search, Retrieval Relevance Tuning
LLM Platforms & Cloud AI Services: Azure OpenAI Service, OpenAI API, Amazon Bedrock, Vertex AI, Vertex AI Agent Builder, AWS SageMaker, Azure Machine Learning, Azure AI Foundry
Machine Learning & Data Science: Classification, Regression, Clustering, Forecasting, Anomaly Detection, Fraud Detection, Recommendation Systems, Feature Engineering, Feature Selection, Model Training, Model Validation, Hyperparameter Tuning, Time Series 
Data Science & Visualization Libraries: NumPy, Pandas, SciPy, Statsmodels, Dask, Matplotlib, Seaborn, Plotly scikit-learn, TensorFlow, Keras, PyTorch, XGBoost, LightGBM, CatBoost, PySpark
NLP & Text Analytics: spaCy, NLTK, BERT, Word2Vec, FastText, Named Entity Recognition (NER), Text Classification, Topic Modeling, Summarization, Information Extraction, Semantic Similarity, Document Understanding
Evaluation, Testing & Observability: LLM Evaluation, RAG Evaluation, Agent Evaluation, RAGAS, LangSmith, Prompt/Response Tracing, OpenAI Agents Tracing, LLM-as-Judge, Groundedness Evaluation, Hallucination Detection, Retrieval Quality Evaluation, Model Evaluation, Regression Testing, Monitoring, Observability, Azure Monitor, Azure Application Insights
AWS Services: S3, SageMaker, Glue, ECS, Textract, QuickSight, Redshift, Amazon Bedrock, Amazon OpenSearch Service, Lambda, 
Azure Services: Azure OpenAI, Azure AI Search, Azure ML, Azure App Service, Azure Kubernetes Service (AKS), Azure Key Vault, Cosmos DB, Azure Monitor, Application Insights, Microsoft Entra ID, Azure RBAC
GCP Services: Vertex AI, Vertex AI Agent Builder, BigQuery, Cloud Storage, Cloud Functions
MLOps, LLMOps & Deployment: Docker, Kubernetes, GitHub Actions, Jenkins, CI/CD, MLflow, Kubeflow, FastAPI, Flask, Microservices, Model Serving, Model Registry, Prompt Versioning, Evaluation Pipelines, Containerized Deployment
Databases, Data Platforms & Warehousing: PostgreSQL, MySQL, SQL Server, MongoDB, Cassandra, Snowflake, BigQuery, Redshift
Security, Governance & Responsible AI: RBAC, Data Governance, Encryption, Secrets Management, Responsible AI Guardrails, PII 
AI-Assisted Development Tools & IDEs: Cursor, GitHub Copilot, JupyterLab, VS Code, PyCharm, IntelliJ IDEA

 
PROFESSIONAL EXPERIENCE

Lead AI/ML Engineer                                                                                                                                                                  Jan 2025 – Present
AMGEN - Thousand Oaks,CA                                                                                                                                                                                   

Summary: Directed the architecture and delivery of an Azure-based multi-agent AI platform for quality investigation workflows at Amgen. The solution supported deviation intake, SOP-grounded analysis, manufacturing and quality evidence discovery, compliance review, persistent case-state orchestration, and reviewer handoff within regulated biopharmaceutical manufacturing operations. Designed as a governed, scalable, and audit-ready decision-support platform, the system improved evidence accessibility, investigation consistency, operational visibility, and cross-functional collaboration across QA, manufacturing, operations, compliance, and technical services teams.

Responsibilities:

•	Developed a coordinated agent architecture consisting of Deviation Intake, Evidence Retrieval, Compliance Review, Investigation Synthesis, and Review Orchestration agents to manage investigation activities across GxP-aligned workflows.
•	Leveraged Azure OpenAI with the OpenAI Agents SDK to enable agent reasoning, tool invocation, structured response generation, agent-to-agent handoffs, execution tracing, and multi-step workflow coordination.
•	Implemented Model Context Protocol integration through MCP servers to provide standardized, controlled connectivity between AI agents and approved enterprise systems, including quality records, case metadata, document repositories, batch references, and workflow actions.
•	Defined MCP tool interfaces with scoped access controls, typed request and response contracts, audit logging, validation rules, and approval checkpoints to ensure safe and traceable interaction with internal systems.
•	Built state-management capabilities for long-running multi-agent investigations, including persistent case context, agent-specific memory boundaries, checkpointing, task status tracking, tool-call history, reviewer comments, and handoff metadata.
•	Created a stateful orchestration layer to preserve and manage investigation context across agents, capturing deviation details, retrieved evidence, compliance observations, unresolved questions, reviewer feedback, and final investigation summaries.
•	Used Azure Cosmos DB and Azure Cache for Redis to store durable case state, temporary orchestration context, workflow checkpoints, agent execution metadata, and resumable investigation session data.
•	Implemented grounded retrieval with Azure AI Search  produced using text-embedding-3-large, allowing agents to retrieve and cite relevant SOPs, batch records, historical deviations, quality procedures, manufacturing documentation.
•	Applied GPT-5.2 for core investigation reasoning, evidence synthesis, compliance-aligned analysis, and investigation summary generation, while using GPT-4o mini for lower-latency tasks such as summarization, classification, routing, and handoff.
•	Combined LLM-as-judge evaluation, deterministic rule-based checks, Azure AI Foundry evaluation capabilities, OpenAI evaluation approaches, and custom Python test harnesses to assess groundedness, relevance, completeness, consistency, compliance alignment, and response quality.
•	Added regression testing for prompts, MCP tools, retrieval settings, structured output schemas, model configurations, and end-to-end multi-agent workflows to reduce quality drift during model, prompt, and pipeline updates.
•	Instrumented agent execution with OpenAI Agents tracing, Azure Application Insights, and Azure Monitor to observe workflow paths, agent handoffs, tool calls, retrieved documents, API latency, dependency health, model usage, runtime exceptions, and operational telemetry.
•	Created CI/CD pipelines with GitHub Actions to automate unit testing, evaluation runs, prompt regression validation, container builds, security scanning, and controlled promotion across development, staging, and production environments before AKS deployment.
•	Secured platform components with Microsoft Entra ID, managed identities, Azure RBAC, private networking patterns, and Azure Key Vault to enforce role-based access, protect credentials, and maintain secure connectivity across enterprise.
•	Implemented human-in-the-loop control points before final investigation conclusions, quality record updates, or downstream workflow actions, ensuring the platform functioned as governed AI decision support rather than autonomous case closure.
•	Increased investigation consistency by requiring AI outputs to be grounded in approved procedures, historical deviation records, batch evidence, and controlled quality documentation instead of unsupported free-form model generation.
•	Standardized quality investigation workflows through reusable agent handoff patterns, MCP-based tool interfaces, structured schemas, persistent state handling, centralized retrieval, and governed access to manufacturing knowledge sources.


SENIOR AI & RAG SPECIALIST                                                                                                                                                Apr 2023 - Dec 2024
Bloomberg L.P, New York City, NY 

Summary: Architected and delivered a GenAI regulatory and financial research copilot at Bloomberg, enabling analysts and tax research professionals to query regulatory guidance, filings, tax content, and financial documents through grounded natural-language search. Designed the platform using RAG techniques including hybrid vector and keyword retrieval, document chunking, metadata filtering, citation-backed response generation, and context-aware prompt assembly to improve answer relevance, traceability, and source grounding. Delivered the solution with Amazon Bedrock for model inference, Amazon OpenSearch Serverless for hybrid retrieval, Amazon Textract for document extraction, and AWS-native deployment, observability, and security controls, incorporating context management to preserve user intent, manage retrieved evidence across multi-turn research sessions, and provide scalable, compliant research support aligned with Bloomberg’s financial research and AI-powered tax workflows.

Responsibilities:
•	Architected a GenAI research copilot for regulatory, tax, filing, and financial document analysis using AWS-native services.
•	Designed a RAG architecture to ground model responses in authoritative source documents and reduce unsupported generation.
•	Implemented hybrid retrieval using Amazon OpenSearch Serverless with vector search, keyword search, metadata filtering, and relevance ranking.
•	Used Cohere Embed English v3 and Cohere Embed Multilingual v3 for high-quality embeddings across regulatory, tax, and financial research content.
•	Developed document chunking strategies to improve retrieval quality across long-form filings, tax guidance, regulatory updates, and research reports.
•	Used Amazon Textract to extract text, tables, and structured content from PDFs, scanned documents, filings, and financial materials.
•	Integrated Amazon Bedrock for managed LLM inference across query understanding, summarization, synthesis, and answer generation.
•	Built context-aware prompt assembly to combine user queries, retrieved passages, source metadata, citations, and conversation history.
•	Implemented multi-turn context management to preserve analyst intent across follow-up questions and evolving research workflows.
•	Added citation-backed response generation to connect answers to specific source passages, filings, regulatory guidance, and tax documents.
•	Designed retrieval pipelines to rank, filter, deduplicate, and select the most relevant evidence within model context-window limits.
•	Built grounding and compliance guardrails to keep responses aligned with retrieved sources and approved research practices.
•	Instrumented the platform with logging, tracing, and monitoring to track retrieval behavior, model latency, errors, and usage patterns.
•	Applied AWS security controls for identity management, encryption, access governance, and protection of sensitive research content.

Data Scientist / MLOps Engineer			 	                                                                                  Feb 2021 – Mar 2023
Exxon Mobile,   Houston TX

Summary: As a Data Scientist focused on MLOps at ExxonMobil, I architected and implemented scalable machine learning pipelines to optimize upstream and downstream oil & gas operations, including predictive maintenance for refinery assets and seismic data analysis. I established the foundational MLOps practices to transition models from prototyping to production, ensuring reliability and performance in critical environments. My role bridged data science and engineering teams to deploy models that enhanced operational efficiency and reduced non-productive time.

Responsibilities:

•	Designed and automated end-to-end ML pipelines for seismic interpretation and predictive maintenance models, reducing equipment downtime by implementing proactive failure alerts for refinery assets.
•	Containerized machine learning models and environments using Docker and orchestrated deployment on Kubernetes clusters to ensure scalable, reliable inference services in production.
•	Implemented CI/CD workflows for ML (GitLab CI/CD) to automate model testing, validation, and deployment, enabling rapid iteration and reducing manual release overhead.
•	Established model versioning, experiment tracking, and model registry protocols using MLflow to ensure reproducibility and governance across all machine learning projects.
•	Developed monitoring dashboards with Prometheus and Grafana to track model performance drift, data drift, and infrastructure health, triggering automated retraining pipelines when needed.
•	Built and optimized large-scale data processing workflows using Apache Spark to handle high-frequency sensor data from drilling rigs and pipeline networks for feature engineering.
•	Leveraged Azure ML to manage cloud-based training workloads and deploy models as scalable endpoints, integrating with existing on-premise data systems.
•	Collaborated with reservoir and drilling engineers to develop models for optimizing well placement and reducing non-productive time through real-time data analysis.
•	Created automated data validation and quality checks to ensure the integrity of input data feeds for mission-critical models in production.
•	Productionized machine learning models, reducing latency by 30% and improving throughput for real-time inference demands in operational settings.
•	Documented MLOps standards and best practices, providing training and guidance to data scientists across teams to elevate the organization's overall ML deployment capabilities.


Data Analyst						                                                                                   Sep 2019 – Feb 2021
Bog Warner, Auburn Hills, MI

Summary: As a Data Analyst at BorgWarner, I supported data-driven decision-making in automotive manufacturing by developing interactive dashboards and reports to monitor supply chain efficiency, production quality, and operational performance. I collaborated with cross-functional teams to translate business requirements into analytical solutions, enabling proactive identification of bottlenecks.

Responsibilities:
•	Developed and maintained Power BI dashboards to track key manufacturing metrics, including production throughput, defect rates, and inventory turnover, enabling real-time monitoring and reducing reporting delays.
•	Wrote complex SQL queries to extract, transform, and analyze data from relational databases, ensuring data accuracy and supporting ad-hoc reporting for supply chain and operations teams.
•	Automated manual reporting processes using Excel VBA and Power Query, reducing weekly reporting time by 15 hours and minimizing human error in data consolidation.
•	Conducted statistical analysis on production data to identify trends and correlations, providing actionable insights to improve quality control and reduce waste in manufacturing processes.
•	Collaborated with engineers and operations managers to define KPIs and design data models that aligned with business goals, enhancing the usability and relevance of analytical outputs.
.
 
EDUCATION

Bachelors in Mechanical Engineering
Master of Business Administration in Analytics- Stevens Institute of Technology, School of Business

CERTIFICATIONS
Google Data Analytics



