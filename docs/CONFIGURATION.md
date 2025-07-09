# Configuration

Retriever uses [pydantic settings management](https://docs.pydantic.dev/latest/concepts/pydantic_settings/). As such, you may configure Retriever in a couple of ways.

## YAML Configs

`/config` contains a variety of yaml files. These files separate Retriever's configuration options into topics, for instance, if you wish to configure the SmartAPI/OpenAPI specification, you'd use `/config/openapi.yaml`. These configuration files will contain comments to explain the purpose and effect of each option.

## Environment Variables

You can also change Retriever configuration options by passing environment variables to the program. The names are case-insensitive, with a configuration file's name (without file suffix) as a prefix to the option. Nested sections are delimited by a double underscore. For instance, if you'd like to change the batch size limit advertized in Retriever's SmartAPI specification, you'd use: `OPENAPI__X_TRAPI__BATCH_SIZE_LIMIT=<number>`

## .env File

A `.env` file placed in the project root directory can also be used in place of explicitly passing in environment variables, and follows the same name rules as above.
