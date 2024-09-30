# coding: utf-8
import os
import urllib.parse

import httpx

from utils.logger import Logger
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

from dotenv import load_dotenv
load_dotenv()  # 加载环境变量
logger = Logger(__name__)
DRUG_KEY = os.environ.get("DRUG_KEY")

def get_medicine_details(medicine_name: str) -> str:
    """Call to get the medicine details."""
    try:
        wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(lang="zh", top_k_results=1))
        return wikipedia.run(medicine_name)
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e}")
        return "Failed to retrieve medicine data"
    except httpx.RequestError as e:
        logger.error(f"Request error occurred: {e}")
        return "Failed to connect to medicine service"
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        return "An error occurred while fetching medicine data"

    # def get_medicine_details(medicine_name: str) -> str:
    # """Call to get the medicine details."""
    # try:
    #     url = "https://apis.tianapi.com/yaopin/index"
    #     params = {'key': DRUG_KEY, 'word': medicine_name}
    #
    #     # Encode the parameters
    #     encoded_params = urllib.parse.urlencode(params)
    #
    #     # Combine the URL with the encoded parameters
    #     full_url = f"{url}?{encoded_params}"
    #
    #     # Set headers for the request
    #     headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    #
    #     # Perform the POST request
    #     with httpx.Client(timeout=10.0) as client:
    #         response = client.post(full_url, headers=headers)
    #         response.raise_for_status()  # Raise an error for bad responses
    #
    #         # Parse the JSON response
    #         data = response.json()
    #         if data.get("code") == 200:
    #             return data["result"]
    #         else:
    #             logger.error(f"No medicine data available for medicine: {medicine_name}")
    #             return "No medicine data available"
    # except httpx.HTTPStatusError as e:
    #     logger.error(f"HTTP error occurred: {e}")
    #     return "Failed to retrieve medicine data"
    # except httpx.RequestError as e:
    #     logger.error(f"Request error occurred: {e}")
    #     return "Failed to connect to medicine service"
    # except Exception as e:
    #     logger.error(f"An unexpected error occurred: {e}")
    #     return "An error occurred while fetching medicine data"


if __name__ == '__main__':
    result = get_medicine_details("阿奇霉素")
    print(result)