

### get mapping
curl -X GET "localhost:9200/enron/_mapping?pretty"  

``` 
curl -X GET "localhost:9200/_mget?pretty" -H 'Content-Type: application/json' -d' 
{
    "docs" : [
        {
            "_index" : "twitter",
            "_id" : "1"
        },
        {
            "_index" : "twitter",
            "_id" : "2"
        }
    ]
}
'
```
```
curl -X PUT "localhost:9200/index?pretty" -H 'Content-Type: application/json' -d'
{
    "settings" : {
        "index" : {
            "similarity" : {
              "my_similarity" : {
                "type" : "BM25"
              }
            }
        }
    }
}
'
```

### get settings:
`curl -X GET "localhost:9200/enron/_settings?pretty"`

```
curl -XGET "localhost:9200/enron/_search?pretty=true" -H 'Content-Type: application/json' -d'
{
    "query": {
       "fuzzy" : { "description" : "smoothie" }
    }
} 
```


curl -X GET "localhost:9200/enron/_search?pretty" -H 'Content-Type: application/json' -d'
{
  "query": { "match": { "body": "stock" } }
}
'

### nested query 
This nested query will all mails, which contain the word stock. The complete mails with all their text units are returned.  
This is not what I wanted, but it might proof useful nonetheless.
```
curl -X GET "localhost:9200/enron/_search?pretty" -H 'Content-Type: application/json' -d'
{
    "query":  {
        "nested" : {
            "path" : "body",
            "query" : {
                 "match" : {"body.content" : "electrons" }
            }
        }
    },
}
'
```


### nested query which only returns the inner document with the hit bam bam:


curl -X GET "localhost:9200/enron/_search?pretty" -H 'Content-Type: application/json' -d'
{
  "_source": {
    "includes": [ "*" ],
    "excludes": [ "body" ]
  },
  "query": {
    "nested": {
      "path": "body",
      "inner_hits": {       

      },
      "query": {
        "constant_score" : {
            "filter": {
                "match": {
                    "body.content" : "electrons" 
                }
            }
        }
      }
    }
  }
}
'





curl -X GET "localhost:9200/twitter/_search?pretty" -H 'Content-Type: application/json' -d'
{
    "query": {
        "match": {
            "message": "elasticsearch"
        }
    },
    "collapse" : {
        "field" : "user" 
    },
    "sort": ["likes"], 
    "from": 10 
}
'
