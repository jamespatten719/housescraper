import requests

def loadJsonResponse(url):
    return json.loads(req.urlopen(url).read())['result']

response = urllib2.urlopen('https://api.instagram.com/v1/tags/pizza/media/XXXXXX')
data = json.load(response)   
print data


def queryPostcode(postcode):
    url = 'https://api.postcodes.io/postcodes?q={}'.format(postcode)
    return loadJsonResponse(url)

def queryPostcode1(postcode):
    url = 'https://api.postcodes.io/postcodes?q=[query]'.format(postcode)
    return loadJsonResponse(url)



print(queryPostcode1('BT146QE'))
print(queryPostcode('BT14 6QE'))



results = requests.get("http://www.bing.com/search", 
              params={'q': query, 'first': page}, 
              headers={'User-Agent': user_agent})

areas = []
postcodes = ['HA1 4LZ', 'BT14 GQE']
for postcode in postcodes:
    area = (((requests.get('https://api.postcodes.io/postcodes?q=' + postcode)).json())['result'])[0]['admin_district']
    areas.append(area)
    query = 'https://api.postcodes.io/postcodes?q='
    
    admin_district