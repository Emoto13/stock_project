import requests

class StockDataFetcher:
    def __init__(self, url:str = "", querystring:dict = {}, rapidapi_key:str = "", rapidapi_host:str = "") -> None:
        self.url = url
        self.querystring = querystring
        self.rapidapi_key = rapidapi_key
        self.rapidapi_host = rapidapi_host
        
    def fetch_data(self):
        map_url_to_function = {
            'https://alpha-vantage.p.rapidapi.com/query': self.fetch_data_alpha_vantage
        }
        return map_url_to_function[self.url]()        

    def fetch_data_alpha_vantage(self):
        headers = {
        'x-rapidapi-key': self.rapidapi_key,
        'x-rapidapi-host': self.rapidapi_host
        }
        response = requests.get(self.url, headers=headers, params=self.querystring)
        return response.json()

