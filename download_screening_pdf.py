"""

Download the gecko driver to control your firefox:
https://github.com/mozilla/geckodriver/releases
and put the "geckodriver" file into the same folder as this script.

Create a file with your user credentials:
yorku_login.yaml:
username: myusername
password: mypassword

"""

import urllib
import requests

url = "https://carex.uber.space/tmp/screen_mail.php?authtoken=xx508xx63817x752xx74004x30705xx92x58349x5x78f5xx34xxxxx51"

values = {'from': 'richard.gerum@yahoo.de',
          'to': 'richard.gerum@protonmail.com',
          'subject': "Screening pdf for today",
          'body': "Please find the attached screening pdf.",
}

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
import os
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from pathlib import Path
from selenium.webdriver.firefox.options import Options
from selenium.common.exceptions import ElementNotInteractableException
from selenium.webdriver.remote.webelement import WebElement
from selenium.webdriver.firefox.firefox_profile import FirefoxProfile

def wrap_retry(func):
    def f(*args, **kwargs):
        for i in range(100):
            try:
                func(*args, **kwargs)
            except ElementNotInteractableException:
                time.sleep(0.01)
                continue
            else:
                break
    return f

WebElement.send_keys = wrap_retry(WebElement.send_keys)



# JavaScript: HTML5 File drop
# source            : https://gist.github.com/florentbr/0eff8b785e85e93ecc3ce500169bd676
# param1 WebElement : Drop area element
# param2 Double     : Optional - Drop offset x relative to the top/left corner of the drop area. Center if 0.
# param3 Double     : Optional - Drop offset y relative to the top/left corner of the drop area. Center if 0.
# return WebElement : File input
JS_DROP_FILES = "var k=arguments,d=k[0],g=k[1],c=k[2],m=d.ownerDocument||document;for(var e=0;;){var f=d.getBoundingClientRect(),b=f.left+(g||(f.width/2)),a=f.top+(c||(f.height/2)),h=m.elementFromPoint(b,a);if(h&&d.contains(h)){break}if(++e>1){var j=new Error('Element not interactable');j.code=15;throw j}d.scrollIntoView({behavior:'instant',block:'center',inline:'center'})}var l=m.createElement('INPUT');l.setAttribute('type','file');l.setAttribute('multiple','');l.setAttribute('style','position:fixed;z-index:2147483647;left:0;top:0;');l.onchange=function(q){l.parentElement.removeChild(l);q.stopPropagation();var r={constructor:DataTransfer,effectAllowed:'all',dropEffect:'none',types:['Files'],files:l.files,setData:function u(){},getData:function o(){},clearData:function s(){},setDragImage:function i(){}};if(window.DataTransferItemList){r.items=Object.setPrototypeOf(Array.prototype.map.call(l.files,function(x){return{constructor:DataTransferItem,kind:'file',type:x.type,getAsFile:function v(){return x},getAsString:function y(A){var z=new FileReader();z.onload=function(B){A(B.target.result)};z.readAsText(x)},webkitGetAsEntry:function w(){return{constructor:FileSystemFileEntry,name:x.name,fullPath:'/'+x.name,isFile:true,isDirectory:false,file:function z(A){A(x)}}}}}),{constructor:DataTransferItemList,add:function t(){},clear:function p(){},remove:function n(){}})}['dragenter','dragover','drop'].forEach(function(v){var w=m.createEvent('DragEvent');w.initMouseEvent(v,true,true,m.defaultView,0,0,0,b,a,false,false,false,false,0,null);Object.setPrototypeOf(w,null);w.dataTransfer=r;Object.setPrototypeOf(w,DragEvent.prototype);h.dispatchEvent(w)})};m.documentElement.appendChild(l);l.getBoundingClientRect();return l"


def drop_files(element, files, offsetX=0, offsetY=0):
    driver = element.parent
    isLocal = not driver._is_remote or '127.0.0.1' in driver.command_executor._url
    paths = []

    # ensure files are present, and upload to the remote server if session is remote
    for file in (files if isinstance(files, list) else [files]):
        if not os.path.isfile(file):
            raise FileNotFoundError(file)
        paths.append(file if isLocal else element._upload(file))

    value = '\n'.join(paths)
    elm_input = driver.execute_script(JS_DROP_FILES, element, offsetX, offsetY)
    elm_input._execute('sendKeysToElement', {'value': [value], 'text': value})


WebElement.drop_files = drop_files

""" configuration """
# output folder
target_folder = Path(os.getcwd()) / "pdf"
target_folder.mkdir(exist_ok=True)
# whether to show the browser window or not
use_headless = False
""" """

# get current files
files = {x for x in target_folder.iterdir() if x.is_file()}

login = {}
if Path("yorku_login.yaml").exists():
    print("encrypt")
    with open("yorku_login.yaml", "r") as fp:
        with open("yorku_login.yaml.entrypted", "wb") as fp2:
            for line in fp:
                for char in line:
                    fp2.write((str(255-ord(char))+"\n").encode())

if Path("yorku_login.yaml.entrypted").exists():
    total_text = ""
    with open("yorku_login.yaml.entrypted", "rb") as fp:
        for line in fp:
            total_text += chr(255-int(line))

    for line in total_text.split("\n"):
        line = line.strip()
        if line == "":
            continue
        key, value = [s.strip() for s in line.split(":")]
        login[key] = value

import time
options = Options()
options.set_preference("browser.download.folderList", 2)
options.set_preference("browser.download.dir", str(target_folder))
options.set_preference("browser.download.useDownloadDir", True)
options.set_preference("browser.helperApps.neverAsk.saveToDisk", "application/pdf")
options.set_preference("pdfjs.disabled", True)
Path("selenium").mkdir(exist_ok=True)
firefox_profile = FirefoxProfile("selenium")
#firefox_profile.set_preference("javascript.enabled", False)
options.profile = firefox_profile
if use_headless is True:
    options.headless = True

os.environ["PATH"] += ":"+os.getcwd()
print("path", os.environ["PATH"])
print("start driver")
driver = webdriver.Firefox(options=options)

def getScreeningPDF(driver):
    driver.get("https://yorku.ubixhealth.com/login")

    driver.find_element(By.TAG_NAME, 'button').send_keys(Keys.RETURN)

    #assert "Python" in driver.title
    print("get login")
    #driver.get("https://yorku.ubixhealth.com/doSaml")
    print(driver.title)

    element = WebDriverWait(driver, 10).until(EC.title_is("Passport York Login"))

    print(driver.title)
    print("input")
    element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "mli")))
    driver.find_element(By.ID, 'mli').send_keys(login["username"])
    driver.find_element(By.ID, 'password').send_keys(login["password"])
    driver.find_element(By.NAME, 'dologin').send_keys(Keys.RETURN)

    element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.ID, "duo_iframe")))

    driver.switch_to.frame("duo_iframe")
    element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "button")))
    while True:
        try:
            driver.find_element(By.TAG_NAME, "button").send_keys(Keys.RETURN)
        except:
            continue
        else:
            break
    driver.switch_to.default_content()

    element = WebDriverWait(driver, 10*60).until(EC.title_is("UBIX Health Screening"))
    element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "new-self-assessment")))

    driver.find_element(By.CLASS_NAME, "new-self-assessment").send_keys(Keys.RETURN)

    element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "btn-continue")))

    from selenium.common.exceptions import NoSuchElementException
    while True:
        try:
            try:
                driver.find_element(By.ID, "choice-14")
            except NoSuchElementException:
                driver.find_element(By.CLASS_NAME, "btn-continue").send_keys(Keys.RETURN)
                element = WebDriverWait(driver, 0.1).until(EC.presence_of_element_located((By.ID, "choice-14")))
            else:
                break
        except:
            continue
        else:
            break

    driver.find_element(By.ID, "choice-14").find_element(By.XPATH, "./..").click()
    driver.find_element(By.CLASS_NAME, "btn-continue").send_keys(Keys.RETURN)

    # Has a doctor, health care provider, or public health unit told you that you should currently be isolating (staying at home)?
    element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "button[value='30']")))
    driver.find_element(By.CSS_SELECTOR, "button[value='30']").send_keys(Keys.RETURN)

    # In the last 10 days, have you been identified as a “close contact” of someone who currently has COVID-19?
    element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "button[value='34']")))
    driver.find_element(By.CSS_SELECTOR, "button[value='34']").send_keys(Keys.RETURN)
    # In the last 10 days, have you received a COVID Alert exposure notification on your cell phone?
    element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "button[value='38']")))
    driver.find_element(By.CSS_SELECTOR, "button[value='38']").send_keys(Keys.RETURN)
    # In the last 14 days, have you travelled outside of Canada and been told to quarantine (per the federal quarantine requirements)?
    element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "button[value='42']")))
    driver.find_element(By.CSS_SELECTOR, "button[value='42']").send_keys(Keys.RETURN)
    # In the last 10 days, have you tested positive on a rapid antigen test or home-based self-testing kit?
    element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "button[value='46']")))
    driver.find_element(By.CSS_SELECTOR, "button[value='46']").send_keys(Keys.RETURN)
    # Is anyone you live with currently experiencing any new COVID-19 symptoms and/or waiting for test results after experiencing symptoms?
    element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "button[value='1007']")))
    driver.find_element(By.CSS_SELECTOR, "button[value='1007']").send_keys(Keys.RETURN)

    element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CLASS_NAME, "o-result-container")))
    driver.find_element(By.CLASS_NAME, "card-body").find_element(By.CSS_SELECTOR, "a[tabindex='0']").send_keys(Keys.RETURN)

    # get new pdf file
    new_filename = list({x for x in target_folder.iterdir() if x.is_file()} ^ set(list(files)[:1]))[0]

    return new_filename

new_filename = getScreeningPDF(driver)
#new_filename = "pdf/screening_100000323461.pdf"#getScreeningPDF(driver)
if 0:
    driver.get("https://outlook.office.com/mail/inbox")
    element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type=email]")))
    driver.find_element(By.CSS_SELECTOR, "input[type=email]").send_keys(login["username"]+"@yorku.ca")
    driver.find_element(By.CSS_SELECTOR, "input[type=submit]").send_keys(Keys.RETURN)

    element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[type=password]")))
    driver.find_element(By.CSS_SELECTOR, "input[type=password]").send_keys(login["password"])

    while True:
        try:
            driver.find_element(By.CSS_SELECTOR, "input[type=submit]").send_keys(Keys.RETURN)
            element = WebDriverWait(driver, 10).until(EC.title_is("Two-Factor Authentication"))
        except Exception:
            continue
        else:
            break

    #driver.switch_to.frame("duo_iframe")
    element = WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.TAG_NAME, "button"))).send_keys(Keys.RETURN)

    # Yes login automatically
    WebDriverWait(driver, 10*60).until(EC.presence_of_element_located((By.CSS_SELECTOR, "input[value=Yes]"))).send_keys(Keys.RETURN)

    #
    #while True:
    #    try:
    WebDriverWait(driver, 10).until(EC.presence_of_element_located((By.CSS_SELECTOR, "#id__7"))).click()
    driver.find_element(By.CSS_SELECTOR, ".div[aria-label=To]").send_keys(values["to"])
    #    except:
    #        continue
    #    else:
    #        break

    driver.find_element(By.CSS_SELECTOR, "div[aria-label=Add a subject]").send_keys(values["subject"])
    driver.find_element(By.CSS_SELECTOR, "div[aria-label=Message Body]").send_keys(values["body"])

    #driver.find_element(By.CSS_SELECTOR, "button[name=Attach]").click()
    #driver.find_element(By.CSS_SELECTOR, "button[name=\"Browse this computer\"]").click()

    # drag and drop the attachment into the body
    driver.find_element(By.CSS_SELECTOR, "div[aria-label=Message Body").drop_files(str(new_filename))

    driver.find_element(By.CSS_SELECTOR, "button[aria-label=Send").send_keys(Keys.RETURN)


driver.quit()

# get new pdf file
#new_filename = list({x for x in target_folder.iterdir() if x.is_file()} ^ set(list(files)[:1]))[0]

files = {'file': open(new_filename, 'rb')}
r = requests.post(url, files=files, data=values)

