import base64
import json
import urllib
import requests


class Recognition:
    def __init__(self):
        self.API_KEY = "G7yhigWMGDHlcK9cs9aBQhkj"
        self.SECRET_KEY = "WNcECzIV1hvLVBXghWFwZOI6FyqfWlFZ"

    def get_access_token(self):
        url = "https://aip.baidubce.com/oauth/2.0/token"
        params = {"grant_type": "client_credentials", "client_id": self.API_KEY, "client_secret": self.SECRET_KEY}
        return str(requests.post(url, params=params).json().get("access_token"))

    @staticmethod
    def get_file_content_as_base64(path, urlencoded=True):
        with open(path, "rb") as f:
            content = base64.b64encode(f.read()).decode("utf8")
            if urlencoded:
                content = urllib.parse.quote_plus(content)
        return content

    def process_image(self, path):
        url = "https://aip.baidubce.com/rest/2.0/ocr/v1/accurate_basic?access_token=" + self.get_access_token()

        image = 'image=' + self.get_file_content_as_base64(
            path) + '&detect_direction=false&paragraph=false&probability=false'
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Accept': 'application/json'
        }
        response = requests.request("POST", url, headers=headers, data=image)
        dic = json.loads(response.text)
        if "words_result" in dic:
            if dic["words_result"]:
                words = dic["words_result"][0]['words']
                return words


if __name__ == '__main__':
    sb = Recognition()
    print(sb.process_image("C:\\Users\\14485\\Pictures\\temp_image\\0\\text_0.png"))
