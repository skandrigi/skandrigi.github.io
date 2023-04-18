from flask import Flask, request, render_template
from selenium import webdriver
from selenium.webdriver.chrome.options import Options as ChromeOptions

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form['ticker']
        url = f'https://finance.yahoo.com/quote/{ticker}/financials?p={ticker}'
        options = ChromeOptions()
        options.headless = True
        driver = webdriver.Chrome(options=options)
        driver.get(url)
        revenue = driver.find_element_by_xpath('//span[text()="Total Revenue"]/following-sibling::span').text
        net_income = driver.find_element_by_xpath('//span[text()="Net Income"]/following-sibling::span').text
        eps = driver.find_element_by_xpath('//span[text()="EPS (Basic)"]/following-sibling::span').text
        driver.quit()
        return render_template('results.html', ticker=ticker, revenue=revenue, net_income=net_income, eps=eps)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=False)
