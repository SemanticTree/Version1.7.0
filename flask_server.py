from flask import Flask, request, render_template
import main
import json
import requests
import urllib


app = Flask(__name__)
json_data = None
input_text = None


@app.route('/')
def simple_page():
    return "Please go to <a href='/input_page'> INPUT PAGE</a>"


@app.route('/input_page')
def input_page():
    return render_template('new-input.html')


@app.route('/wiki_text')
def wiki_test():
    base_url = "https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&exintro&explaintext&redirects=1&exsentences=3&titles="
    input_text = request.args.get("url")
    parse_title = urllib.parse.quote(input_text)
    url = base_url + parse_title
    R = requests.get(url)
    ex = list(R.json()['query']['pages'].values())[0]['extract']
    print(ex)
    return "" + ex


@app.route('/get_cytoscape_json_for_text', methods=['POST', 'GET'])
def get_cytoscape_json_for_text():
    global json_data, input_text
    if request.method == "POST":
        input_text = request.form["input-text"]
        return render_template('new-output.html')


@app.route('/juggad')
def juggad():
    global json_data, input_text
    threshold_value = request.args.get('threshhold_value')
    max_connections_value = request.args.get('max_connections_value')
    print(threshold_value, max_connections_value)
    if threshold_value and max_connections_value:
        json_data = json.dumps(
            main.generate_structured_data_from_text(input_text,
                                                    threshold_value=float(threshold_value),
                                                    max_connections_value=int(max_connections_value)))
    else:
        json_data = json.dumps(main.generate_structured_data_from_text(input_text))
    return json_data + "|split|" + input_text


if __name__ == '__main__':
    app.run(debug=True)
