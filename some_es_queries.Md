

### get mapping
curl -X GET "localhost:9200/enron/_mapping?pretty"  


### get some documents via id
``` 
curl -X GET "localhost:9200/_mget?pretty" -H 'Content-Type: application/json' -d' 
{
    "docs" : [
        {
            "_index" : "enron",
            "_id" : "1"
        },
        {
            "_index" : "enron",
            "_id" : "2"
        }
    ]
}
'
```

### set the similarity to BM25 
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
       "more_like_this" : {
           "fields" : ["body.content"],
           "like" : ["the stock market is fucked"],
           "min_term_freq" : 1,
           "min_doc_freq":1
           
       }
   }
    }
  },
  "size" : 0
}
'

### query which returns the inner documents which cause the hit of the document.

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
            "match": {
                "body.content" : "electrons" 
            }
      }
    }
  }
}
'




### more like this query which returns the inner documents which cause the hit of the document.

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
          "more_like_this" : {
              "fields" : ["content"],
              "like" : "You owe me dinner"
          }
      }
    }
  },
  "size" : 0
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


curl -X GET "localhost:9200/_search?pretty" -H 'Content-Type: application/json' -d'
{
    "query": {
        "more_like_this" : {
            "fields" : ["body"],
            "like" : "you owe me dinner"
        }
    }
}
'


### nested query with more like this
This nested query will all mails, which contain the word stock. The complete mails with all their text units are returned.  
This is not what I wanted, but it might proof useful nonetheless.
```
curl -X GET "localhost:9200/enron/_search?pretty" -H 'Content-Type: application/json' -d'
{
    "query":  {
      "nested" : {
        "path" : "body",
        "query" : {
          "more_like_this" : {
            "fields": ["body.content"],
            "like" : "you owe me dinner" 
            "min_term_freq" : 1
          }
        }
      }
    }
}
'
```