# Changelog

All notable changes to this project will be documented in this file. See [standard-version](https://github.com/conventional-changelog/standard-version) for commit guidelines.

## [1.18.0](https://github.com/biothings/bte_trapi_query_graph_handler/compare/v1.17.6...v1.18.0) (2021-08-20)


### Features

* add caching param in cache handler ([bfaa179](https://github.com/biothings/bte_trapi_query_graph_handler/commit/bfaa17908a0c7ef584b146390659186fe06896dc))
* add workflow validation for trapi v1.2 ([76a4164](https://github.com/biothings/bte_trapi_query_graph_handler/commit/76a4164bcacfc1a4c9dc74fb5d4024ece7ef0e08))


### Bug Fixes

* :bug: check if cachedQueryResults is empty ([062d931](https://github.com/biothings/bte_trapi_query_graph_handler/commit/062d931bcc9d51e966fb65e470bd1897588ee312))
* :bug: set a default caching expiration ([173f34b](https://github.com/biothings/bte_trapi_query_graph_handler/commit/173f34b51b7a033e4a4a77dae766473ae45d5d6e))
* biolink v2.1 related fixes ([ff15564](https://github.com/biothings/bte_trapi_query_graph_handler/commit/ff155648163f74ef209ed71666bce860fb5c7f26))
* cache extra properties just in case ([344aabd](https://github.com/biothings/bte_trapi_query_graph_handler/commit/344aabd2037e172bd3f3970db1e31f0ac7ce00c6))
* clone records recursively for caching ([b7a71c4](https://github.com/biothings/bte_trapi_query_graph_handler/commit/b7a71c49c7e29f124b87b6b73e2eed747844aa13))
* code line length reformat ([7000a0b](https://github.com/biothings/bte_trapi_query_graph_handler/commit/7000a0b7eaaa0539648e2d6e231f8e871b941dd6))
* correctly cache semanticType ([7d21651](https://github.com/biothings/bte_trapi_query_graph_handler/commit/7d21651fc013154d9704e22e89be800f968f3c08))
* input->output for output node ([aa4a529](https://github.com/biothings/bte_trapi_query_graph_handler/commit/aa4a529a597efc5c33fddbed6796527e2ba7438d))
* issue [#164](https://github.com/biothings/bte_trapi_query_graph_handler/issues/164). add score. rm unused methods. ([57b4c4b](https://github.com/biothings/bte_trapi_query_graph_handler/commit/57b4c4b756100bcd6faa40c83bc4f637cb752c79))
* tests pass for issue [#164](https://github.com/biothings/bte_trapi_query_graph_handler/issues/164) ([8dfb3ca](https://github.com/biothings/bte_trapi_query_graph_handler/commit/8dfb3cabdee3e052793d503c7bf4eebddd99601e))
* WIP for https://github.com/biothings/BioThings_Explorer_TRAPI/issues/164 ([97963b6](https://github.com/biothings/bte_trapi_query_graph_handler/commit/97963b6f3379de4f1ed7ff8e5bedbcd303f34271))

### [1.17.6](https://github.com/biothings/bte_trapi_query_graph_handler/compare/v1.17.5...v1.17.6) (2021-06-24)


### Bug Fixes

* :bug: fix test query ([00f84b0](https://github.com/biothings/bte_trapi_query_graph_handler/commit/00f84b05611667a2ad282087589865314cf13f83))
* :bug: fix tests array value, check item exists error fix, skip test returning with 0 results temp ([ef1e1a9](https://github.com/biothings/bte_trapi_query_graph_handler/commit/ef1e1a97dd4c8051ccc9611d2b89b4d92b22ab1a))
* :sparkles: fix repeating predicate restriction logs ([4b10df0](https://github.com/biothings/bte_trapi_query_graph_handler/commit/4b10df0a4f0af169b4cefe5435b2a0d0f9dd5f43))
* check record exists before adding to results ([b0ec67c](https://github.com/biothings/bte_trapi_query_graph_handler/commit/b0ec67cf73202050856be01f6024863dc618f6df))
* revert last, not issue ([636ac1a](https://github.com/biothings/bte_trapi_query_graph_handler/commit/636ac1ab5a4843d3d1aa44223115a4962288faa6))

### [1.17.5](https://github.com/biothings/bte_trapi_query_graph_handler/compare/v1.17.4...v1.17.5) (2021-06-17)

### [1.17.4](https://github.com/biothings/bte_trapi_query_graph_handler/compare/v1.17.3...v1.17.4) (2021-06-16)


### Bug Fixes

* fix small syntax req ([8787893](https://github.com/biothings/bte_trapi_query_graph_handler/commit/87878936ce00b4bd3893243e65e0bdeb778660a2))
* remove long log ([742871e](https://github.com/biothings/bte_trapi_query_graph_handler/commit/742871ecea670795d133270f31b1b68ad86b3d01))
* remove repeating output log ([49f25ee](https://github.com/biothings/bte_trapi_query_graph_handler/commit/49f25eeeffb24f73c5d9fc81bb0a400ff5cc02f4))

### [1.17.3](https://github.com/biothings/bte_trapi_query_graph_handler/compare/v1.17.2...v1.17.3) (2021-06-15)


### Bug Fixes

* :bug: post query filter assumed predicate edge metadata would always be present, fix to resume normally if not found ([a922134](https://github.com/biothings/bte_trapi_query_graph_handler/commit/a922134a184e54a9554e51027719767252d1dc61))
* :bug: TRAPI 1.1 response validation fix ([ef2ac87](https://github.com/biothings/bte_trapi_query_graph_handler/commit/ef2ac87612dd663045668a7a37d2239d163601f9))
* Add check for meta_knowledge_graph paths conatining 1.1 prefix, add default score of 1 to all queries temporarily, add post query logs to results ([56382e3](https://github.com/biothings/bte_trapi_query_graph_handler/commit/56382e37fdada35094b1541f51d76ed34e1aefba))
* fix conflict ([6cb4767](https://github.com/biothings/bte_trapi_query_graph_handler/commit/6cb4767ac630ea16e3727be8e28cacddee3363f1))
* update repo owner in package specs ([85cce99](https://github.com/biothings/bte_trapi_query_graph_handler/commit/85cce9917e6b8bef78a01876f3157e535d95df4a))

### [1.17.2](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.17.1...v1.17.2) (2021-06-15)


### Bug Fixes

* :bug: change type to value_type_id in TRAPI response ([4f34d5f](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/4f34d5f922c1914d550a1a9a22662f73282424fd))

### [1.17.1](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.17.0...v1.17.1) (2021-06-04)


### Bug Fixes

* upgrade call-apis pkg ([98bde77](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/98bde7755ea9df73c0d45cd8f2b4b24638932ce7))

## [1.17.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.16.2...v1.17.0) (2021-06-04)

### [1.16.2](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.16.1...v1.16.2) (2021-06-04)


### Features

* edge class includes expanded_predicates attribute ([774e4f9](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/774e4f956f521f4cd725322be6bcc50034168494))
* post query filter by predicates ([7d0e2c7](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/7d0e2c709fdda09bb333dc5a867be67f6a65eeb3))


### Bug Fixes

* :bug: add attribute_type_id in output nodes and edges ([0ff6c0b](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/0ff6c0b4e03177d3032ec703f03bd5f9bcbad272))

### [1.16.1](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.16.0...v1.16.1) (2021-05-12)

## [1.16.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.15.0...v1.16.0) (2021-05-11)


### Features

* :sparkles: add Trapi 1.1 support ([32361d7](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/32361d77decec18fd700f9dc669241755640df89))


### Bug Fixes

* :bug: fix query graph handling error ([6dd3a4b](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/6dd3a4baabefeba62276b95f6c3592ce8a4f71b6))

## [1.1.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.0.1...v1.1.0) (2021-03-24)

## [1.15.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.14.0...v1.15.0) (2021-05-04)

## [1.14.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.13.1...v1.14.0) (2021-04-29)


### Features

* :sparkles: support expanding predicates based on biolink hierarchy ([6345571](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/6345571e60998a403c946aabed2fc4a07d8ebbdd))

### [1.13.1](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.13.0...v1.13.1) (2021-04-29)

## [1.13.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.12.1...v1.13.0) (2021-04-28)


### Features

* :sparkles: expand predicate based on biolink model ([c26b317](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/c26b317d5b70dbe2ffc18dd1a7c324a2e13f581e))

### [1.12.1](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.12.0...v1.12.1) (2021-04-27)

## [1.12.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.11.0...v1.12.0) (2021-04-27)

## [1.11.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.10.0...v1.11.0) (2021-04-27)


### Bug Fixes

* :bug: change NCBIGENE to NCBIGene, add RHEA ([fbe4735](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/fbe473580e5a2a2611a9b723eac661bf6e1bb60b))

## [1.10.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.9.0...v1.10.0) (2021-04-23)

## [1.9.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.8.0...v1.9.0) (2021-04-23)


### Features

* :sparkles: support biocarta pathway id ([011134c](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/011134c188ccddfff2b0c5054fc2a769d424308f))

## [1.8.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.7.0...v1.8.0) (2021-04-20)


### Features

* :sparkles: set a 3s timeout for all TRAPI APIs ([87f412e](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/87f412eae73ef53f2ef118badad20daef5c9c615))
* :sparkles: set a 3s timeout for all TRAPI APIs ([4d1c3d4](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/4d1c3d49e61a446ff76cf7106671dbda794ea0a3))

## [1.7.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.6.0...v1.7.0) (2021-04-02)


### Features

* :sparkles: support limiting ops from /predicates to user specified api names ([437321e](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/437321e6f654627043a677e7d1268d169706b9b5))

## [1.6.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.5.1...v1.6.0) (2021-04-02)


### Features

* support using providing a list of APIs to include ([24d4531](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/24d45319dbe2eb12496d129517e0c3f977d8fad2))

### [1.5.1](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.5.0...v1.5.1) (2021-04-01)


### Features

* :sparkles: allow user to specify if they want to include reasoner apis ([b89c288](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/b89c288e0b6f72523d2f6e1c81af0468523b06d6))

## [1.5.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.4.0...v1.5.0) (2021-04-01)


### Bug Fixes

* :bug: should check for predidcatesPath param instead of smartapiPath ([ac413e6](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/ac413e6185cff37ed80eb76a5de1ae7cc1352d12))

## [1.4.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.3.1...v1.4.0) (2021-03-31)


### Features

* :sparkles: include TRAPI APIs when constructing meta-kg ([07688cb](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/07688cb93ebe13be61dd041d2c155a478dac5d5b))
* :sparkles: restrict to a list of APIs to use in BTE ([defdd65](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/defdd656f30a04af85e37ef525d05ec247cc659b))
* :sparkles: support constructing TRAPI based bte edge ([02afc54](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/02afc54bb71492cba9c0a311aef27a416798a8fa))
* :sparkles: support trapi quer ([1658ad2](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/1658ad2f77cbb8c41dba72c871bca595a4495d2f))

### [1.3.1](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.3.0...v1.3.1) (2021-03-29)

## [1.3.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.2.2...v1.3.0) (2021-03-29)


### Features

* :sparkles: upgrade to biolink 1.7.0 ([db58226](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/db58226e602e68f1eb638bc3221035a0c873b4e2))

### [1.2.2](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.2.1...v1.2.2) (2021-03-29)

### [1.2.1](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.2.0...v1.2.1) (2021-03-26)


### Bug Fixes

* :bug: fix edge_metadata deepcopy issue ([9e34876](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/9e3487693abd220f0f453ccc6d60c500a44e3caf))

## [1.2.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.1.10...v1.2.0) (2021-03-26)


### Bug Fixes

* :bug: fix typo in npm release script ([2b80cd3](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/2b80cd330032bc3539e2c494e69f238bf2e71eca))

### [1.1.10](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.1.9...v1.1.10) (2021-03-26)

### [1.1.9](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.1.8...v1.1.9) (2021-03-26)


### Bug Fixes

* :bug: fix wrong attribute type ([2b8d57b](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/2b8d57b70df62aad2cbb7d79d9108bfe1f828234))

### [1.1.8](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.1.7...v1.1.8) (2021-03-26)

### [1.1.7](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.1.6...v1.1.7) (2021-03-25)

### [1.1.6](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.1.5...v1.1.6) (2021-03-25)

### [1.1.5](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.1.4...v1.1.5) (2021-03-25)


### Bug Fixes

* :bug: fix getCategories of undefined error ([8d19f74](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/8d19f74f371d8a04dd78de18b95c7b40623e647c))

### [1.1.4](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.1.3...v1.1.4) (2021-03-24)


### Bug Fixes

* :bug: fix wrong edge id in results section ([a31c7eb](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/a31c7eb62005663940558d3be2d854f7debc8f67))

### [1.1.3](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.1.2...v1.1.3) (2021-03-24)


### Bug Fixes

* :bug: fix missing biolink prefix ([0700e71](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/0700e7149f1c9d3a25b7db67dd46d8c9e7d66f2c))

### [1.1.2](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.1.1...v1.1.2) (2021-03-24)


### Bug Fixes

* :bug: fix publications not display ([560f3fd](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/560f3fd5cd9759418de3b06b8b6a728c85984cb7))
* :bug: fix tyypo ([e630909](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/e6309090c6ec432b0af537d8b3471daa21512698))

### [1.0.1](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v0.8.0...v1.0.1) (2021-03-24)

## [0.8.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v1.0.0...v0.8.0) (2021-03-23)


### Features

* :sparkles: add chemicalsubstance node when user specify drug as output ([ffb81b3](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/ffb81b3f9397bd325ca38f3242f5cd110c03b288))

## [1.0.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v0.7.0...v1.0.0) (2021-03-23)

### [1.1.1](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v0.7.0...v1.1.1) (2021-03-24)


### Features

* :sparkles: support query graph with explain type of query ([2d767fd](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/2d767fd67d04c8956e3d55c2438def12b79ec104))


## [0.7.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v0.6.0...v0.7.0) (2021-03-18)


### Features

* :sparkles: add num_of_participants property for pathway ([dc8ceca](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/dc8cecaddcf04a2803189d6ebe7822dabbec36ad))

## [0.6.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v0.5.0...v0.6.0) (2021-03-18)


### Features

* :sparkles: add node attributes for output nodes in kg ([506afbc](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/506afbc749e90c9f8eb2a47ed924f05dba45f1c7))

## [0.5.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v0.4.0...v0.5.0) (2021-03-18)


### Features

* :sparkles: display additional attributes for nodes in kg ([e79dbe6](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/e79dbe608468aadbff14be7da155e4f46a807557))

## [0.4.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v0.3.0...v0.4.0) (2021-03-16)


### Features

* :sparkles: add log for finding smartapi edges ([589e2f8](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/589e2f895744f72daf7b6ffcfe5589cb98a89bec))

## [0.3.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v0.2.3...v0.3.0) (2021-03-16)


### Features

* :sparkles: support traversing biolink model to find descendant classes ([4e00df1](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/4e00df17e4a6fc70be1b219eb33e4458a0c47d73))
* :sparkles: travese biolink hierarchy when handling node category ([017bf6f](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/017bf6ffc73a737c7dca5eac5857768f7a746ae5))

### [0.2.3](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v0.2.2...v0.2.3) (2021-03-16)


### Bug Fixes

* :bug: fix issue regarding output id not correctly updated ([195b40b](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/195b40b387395571f334fdd5dbd5480d8ccc4665))

### [0.2.2](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v0.2.1...v0.2.2) (2021-03-16)


### Bug Fixes

* :bug: fix issue when generating hashed edge id ([63cb257](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/63cb257ec6e07e042a225bf7cd65913d76c3904d))

### [0.2.1](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v0.2.0...v0.2.1) (2021-03-16)

## [0.2.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v0.1.1...v0.2.0) (2021-03-15)


### Features

* :sparkles: support multiple predicates ([be1adea](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/be1adea8c58a7c3ab15cd24771db67cb0df156a5))

### [0.1.1](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v0.1.0...v0.1.1) (2021-03-12)

## [0.1.0](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v0.0.4...v0.1.0) (2021-03-12)


### Features

* :sparkles: modify entry methods ([4f3f2c9](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/4f3f2c9a73f648263144f7d198feddc68baa1217))


### Bug Fixes

* :bug: include local json files in build ([97825e1](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/97825e17a96f7a1178bb5f9768b1ea5d017aa521))

### [0.0.4](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v0.0.3...v0.0.4) (2021-03-12)


### Bug Fixes

* :bug: fix wrong import path  for InValidQueryGraphError ([92115b7](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/92115b7080c6fefae309d27a60291e274a9e403f))

### [0.0.3](https://github.com/kevinxin90/bte_trapi_query_graph_handler/compare/v0.0.2...v0.0.3) (2021-03-12)


### Bug Fixes

* :bug: include biolink.json file as part of the built folder ([1d08e25](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/1d08e2554660e3cf857c26e8ef2e3d0f70800ad8))

### 0.0.2 (2021-03-11)


### Features

* :sparkles: initial implementation of Query Graph Handler ([9ca397d](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/9ca397d99ccdcc98a0b5d6a6d20ef6765d86a9a7))


### Bug Fixes

* :bug: fix typescript error ([343c548](https://github.com/kevinxin90/bte_trapi_query_graph_handler/commit/343c5484127e242186adc3ec5f2bba10c08d0397))
