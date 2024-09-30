# coding: utf-8
import os

import httpx
from langchain_core.messages import AIMessage
from utils.logger import Logger
from langgraph.prebuilt import ToolNode

import pandas as pd
import difflib

from dotenv import load_dotenv
load_dotenv()  # 加载环境变量
logger = Logger(__name__)
AMAP_KEY = os.environ.get("AMAP_KEY")

def get_city_adcode(location):
    # Read the Excel file
    df = pd.read_excel('../AMap_adcode_citycode.xlsx')

    # Create a dictionary to store the mapping between area names and adcodes
    area_adcode_dict = dict(zip(df['中文名'], df['adcode']))

    def fuzzy_match(query, choices, limit=1):
        matches = difflib.get_close_matches(query, choices, n=limit, cutoff=0.8)
        return [(match, difflib.SequenceMatcher(None, query, match).ratio()) for match in matches]

    # Perform fuzzy matching on the input location
    matches = fuzzy_match(location, list(area_adcode_dict.keys()))

    if matches:
        matched_area, score = matches[0]
        return area_adcode_dict[matched_area]
    else:
        return None


def get_weather(location: str) -> str:
    """Call to get the current weather."""
    try:
        city_adcode = get_city_adcode(location)
        if not city_adcode:
            logger.error(f"Invalid city adcode for location: {location}")
            return "Invalid location"


        url = f"https://restapi.amap.com/v3/weather/weatherInfo?city={city_adcode}&key={AMAP_KEY}"
        # Set a timeout for the request
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url)
            response.raise_for_status()  # Raise an error for bad responses

            data = response.json()
            print(data)
            if data.get("status") == "1":
                return data["lives"]
            else:
                logger.error(f"No weather data available for city adcode: {city_adcode}")
                return "No weather data available"
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error occurred: {e}")
        return "Failed to retrieve weather data"
    except httpx.RequestError as e:
        logger.error("Request error occurred: %s", e)
        return "Failed to connect to weather service"
    except Exception as e:
        logger.error("An unexpected error occurred: %s", e)
        return "An error occurred while fetching weather data"



if __name__ == '__main__':
    # message_with_single_tool_call = AIMessage(
    #     content="",
    #     tool_calls=[
    #         {
    #             "name": "get_weather",
    #             "args": {"location": "sf"},
    #             "id": str(uuid.uuid4()),
    #             "type": "tool_call",
    #         }
    #     ],
    # )
    #
    # # Call the tool node directly with the AIMessage
    # result = tool_node.invoke({"messages": [message_with_single_tool_call]})
    # print(result)
    # Example usage
    location = "合肥"  # Replace with the actual location name you want to query
    # adcode = get_city_adcode(location)
    weaher = get_weather(location)
    print(weaher)