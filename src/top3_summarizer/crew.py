from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task, tool
from crewai.agents.agent_builder.base_agent import BaseAgent
from typing import List
from crewai_tools import (
    SerperDevTool,
    WebsiteSearchTool,
    ScrapeWebsiteTool,
)
import os
from dotenv import load_dotenv
from top3_summarizer.tools.sentiment_tool import LLMSentimentTool


load_dotenv()
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class Top3Summarizer():
    """Top3Summarizer crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"
    
    @tool
    def serper_search(self):
        return SerperDevTool()

    @tool
    def web_search(self):
        return WebsiteSearchTool()

    @tool
    def scrape(self):
        return ScrapeWebsiteTool()
    
    @tool
    def llm_sentiment_tool(self):
        return LLMSentimentTool()
    
    @agent
    def searcher(self):
        return Agent(config=self.agents_config["searcher"])
    
    @agent
    def ranker(self):
        return Agent(config=self.agents_config["ranker"])
    @agent
    def reader(self):
        return Agent(config=self.agents_config["reader"])
    
    @agent
    def summarizer(self):
        return Agent(config=self.agents_config["summarizer"])
    
    @agent
    def analyst(self):
        return Agent(config=self.agents_config["analyst"])
    
    @agent
    def editor(self):
        return Agent(config=self.agents_config["editor"])
    
    @task
    def search(self):
        return Task(config=self.tasks_config["search"])
    
    @task
    def rank(self):
        return Task(config=self.tasks_config["rank"])
    
    @task
    def read(self):
        return Task(config=self.tasks_config["read"])
    
    @task
    def summarize(self):
        return Task(config=self.tasks_config["summarize"])
    
    @task
    def analyze_sentiment(self):
        return Task(config=self.tasks_config["analyze_sentiment"])
    
    @task
    def edit(self):
        return Task(config=self.tasks_config["edit"])
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=[self.searcher(), 
                    self.ranker(), 
                    self.reader(), 
                    self.summarizer(),
                    self.analyst(),
                    self.editor()],
            tasks=[self.search(), 
                    self.rank(), 
                    self.read(), 
                    self.summarize(),
                    self.analyze_sentiment(),
                    self.edit()],
            verbose=True
        )