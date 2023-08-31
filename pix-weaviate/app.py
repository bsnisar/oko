import weaviate


client = weaviate.Client("https://some-endpoint.weaviate.network") 



client.schema.get()  # Get the schema to test connection


