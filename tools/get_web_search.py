import httpx


# from langchain_community.tools import DuckDuckGoSearchResults
# from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
#
# wrapper = DuckDuckGoSearchAPIWrapper(time="d",backend='api')
#
# search = DuckDuckGoSearchResults(api_wrapper=wrapper)
#
# result = search.invoke("地舒单抗")
#
# print(result)


from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="zh",top_k_results=1))

result = wikipedia.run("地舒单抗")

print(result)