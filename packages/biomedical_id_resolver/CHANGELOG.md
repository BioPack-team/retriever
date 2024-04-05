# Changelog

All notable changes to this project will be documented in this file. See [standard-version](https://github.com/conventional-changelog/standard-version) for commit guidelines.

## [3.11.0](https://github.com/biothings/biomedical_id_resolver.js/compare/v3.10.0...v3.11.0) (2022-06-20)


### Features

* :unicorn_face: add biothings apis node attributes during id resoltuion ([4d0a345](https://github.com/biothings/biomedical_id_resolver.js/commit/4d0a34527aae0e75dec1f325083e90ec3335bb31))
* add resolver using SRI ([6302a7e](https://github.com/biothings/biomedical_id_resolver.js/commit/6302a7ed21227c241029cfc973e818da14586978))
* identify BTE via User-Agent ([e20d2fb](https://github.com/biothings/biomedical_id_resolver.js/commit/e20d2fb943e192d61e784ee23bde7ba138039c40))
* re-throw SRI resolver errors with more informative msg ([cc266b9](https://github.com/biothings/biomedical_id_resolver.js/commit/cc266b9c0a3a40985f77b4136754f74d8baa265b))
* refactor SRI resolver to minimize queries made ([6783b4e](https://github.com/biothings/biomedical_id_resolver.js/commit/6783b4e1b155cc58cebbb11bad57b50999d33734))


### Bug Fixes

* :bug: check for empty list of semantic types given to attr module ([a865c1c](https://github.com/biothings/biomedical_id_resolver.js/commit/a865c1ca2b9dca32c4550aaa746b33eb834d2730))
* :bug: only the first id of each prefix was returned ([1dd555c](https://github.com/biothings/biomedical_id_resolver.js/commit/1dd555cfda5b18c43b709cc6760a2d24269e64db))
* :sparkles: add attrs to unresolvable entities ([5981e50](https://github.com/biothings/biomedical_id_resolver.js/commit/5981e5076536eb03d0ffa8255b6247add7e1e9c6))
* :sparkles: remove debug logs ([0ee21a2](https://github.com/biothings/biomedical_id_resolver.js/commit/0ee21a2fe2e8ea78adce6b2cde14e9577cab59cb))
* :sparkles: rename chemblmax phase field to FDA approved ([27ce8fd](https://github.com/biothings/biomedical_id_resolver.js/commit/27ce8fdd8f4fd91d93612d7180e60524665c697d))
* :sparkles: restructure add attr module, supports batch, new call signature ([2ad62c9](https://github.com/biothings/biomedical_id_resolver.js/commit/2ad62c9bd4b883a984d14f651f3451fbd37343f1))
* :sparkles: skip tests not relevant to current prod, update relevant to use ChemicalEntity ([aed0a04](https://github.com/biothings/biomedical_id_resolver.js/commit/aed0a049cf31220ef693dc651507d718c2797d0c))
* :sparkles: update location where addattributes gets called, update method signature to allow for list of semantic types vs one ([2954410](https://github.com/biothings/biomedical_id_resolver.js/commit/29544105594bb34a4475e4bee158b8dc933943f6))
* :sparkles: update max_phase mapped to drug_regulatory_status_world_wide ([1032533](https://github.com/biothings/biomedical_id_resolver.js/commit/1032533a21db1a2d5d1e09219b457d8599ba358c))
* add _leafSemanticType and _dbIDs fields ([c9e74bc](https://github.com/biothings/biomedical_id_resolver.js/commit/c9e74bcff2c3dc6bd6f9fed612d046927402d72f))
* add batching to sri query to fix 414 errors ([326fca6](https://github.com/biothings/biomedical_id_resolver.js/commit/326fca6f055f8bb0315075e38c089b8551a99751))
* add names to curies array in sri resolver ([92062b3](https://github.com/biothings/biomedical_id_resolver.js/commit/92062b32958f0cdb39883ce72ffc07196e908302))
* change CLINVAR id from RCV to Variant ID ([53d403a](https://github.com/biothings/biomedical_id_resolver.js/commit/53d403a3f1cda20fafb13df0dfa0c38d445e2a96))
* change id priorities to match biolink's ([aae97c5](https://github.com/biothings/biomedical_id_resolver.js/commit/aae97c596f5d335fc28d7551c3e83bbc722be33d))
* comment out semantic types that aren't used ([347bb82](https://github.com/biothings/biomedical_id_resolver.js/commit/347bb82053336dec5f8d140f8c8e61d886dc3d3d))
* fix problem with unresolved id response shape ([a3c561c](https://github.com/biothings/biomedical_id_resolver.js/commit/a3c561c3dac2b61be50c1e70cd17bffba43dff30))
* fix results merging ([502e4ff](https://github.com/biothings/biomedical_id_resolver.js/commit/502e4ff6ddcbefc84d853cf8b4b01bdcc9e8bf86))
* handle empty responses from node normalizer ([8172b4a](https://github.com/biothings/biomedical_id_resolver.js/commit/8172b4a0748f86e9389175479a1de5ffb56c08a5))
* normalizer queries in 5k chunks ([1c18cc0](https://github.com/biothings/biomedical_id_resolver.js/commit/1c18cc0b2011da8479b64713df2a52b7b15029bd))
* query using post instead of get ([58fffec](https://github.com/biothings/biomedical_id_resolver.js/commit/58fffeccb86763a3e4116877bc6cc825de5fda07))
* remove dbnsfp field mapping for clinvar id namespace ([04ac54c](https://github.com/biothings/biomedical_id_resolver.js/commit/04ac54c97859a00ba0fe06cc016b965fa1575ee5))
* remove names from curies ([942c93e](https://github.com/biothings/biomedical_id_resolver.js/commit/942c93eb95c50de78645a36788ba5df0ff8cd06a))
* return semantic type when curie is unresolvable ([fc5632a](https://github.com/biothings/biomedical_id_resolver.js/commit/fc5632a3e1b0b249788ca2115d54e3e8cf80e40b))
* shorten debug statement ([94e7197](https://github.com/biothings/biomedical_id_resolver.js/commit/94e719763465beb2801728452920bba3a05b6ec2))
* update sri endpoint to prod ([fdc3ebb](https://github.com/biothings/biomedical_id_resolver.js/commit/fdc3ebb1ace120ae2f41c78ef37f752cd273b406))
* update tests with new signature ([5bd7ae0](https://github.com/biothings/biomedical_id_resolver.js/commit/5bd7ae030ffb68fdca68a9ddac26ac61cd194f5d))
* use input semantic type instead of sri resolved type ([5854a77](https://github.com/biothings/biomedical_id_resolver.js/commit/5854a77b2aaa0e0f88ccf274be1f718698636b48))
* use sri resolved type when input type is unknown ([30f465c](https://github.com/biothings/biomedical_id_resolver.js/commit/30f465cc4012a98a9b4b9577cde21c77ef466d65))

## [3.10.0](https://github.com/biothings/biomedical_id_resolver.js/compare/v3.9.1...v3.10.0) (2021-08-16)


### Features

* support Disease NCIT, SNOMEDCT IDs ([9cc4908](https://github.com/biothings/biomedical_id_resolver.js/commit/9cc4908a62fe176212e0f2b23ed0d9ca381b7c83))


### Bug Fixes

* add BiologicalProcess KEGG back in ([ccf257a](https://github.com/biothings/biomedical_id_resolver.js/commit/ccf257a6041d01157d6646113b1ff699145d63f6))
* add ClinicalFinding semantic type ([56f083f](https://github.com/biothings/biomedical_id_resolver.js/commit/56f083ff06406bdb5d8ebc59d0ebe40319928c4b))
* add Disease, SequenceVariant mappings ([e36bc09](https://github.com/biothings/biomedical_id_resolver.js/commit/e36bc091d1f1988441fd96f987829fe146f65de0))
* add MEDDRA id mapping for Disease semantic type ([e914976](https://github.com/biothings/biomedical_id_resolver.js/commit/e9149760a1361389b27d5b674545f897068a16a6))
* add Pathway GO, SMPDB IDs ([c8b2556](https://github.com/biothings/biomedical_id_resolver.js/commit/c8b2556846dcf8d482cfabab85cb39afdf6b37db))
* adjust fields for mapping SNOMEDCT ([88dd2a4](https://github.com/biothings/biomedical_id_resolver.js/commit/88dd2a49021bfc42b22ade2936fb0f56d7a80e80))
* config + types file: ChemicalSubstance -> SmallMolecule, drug semantic type ([2cfbe88](https://github.com/biothings/biomedical_id_resolver.js/commit/2cfbe889bc32d070a2712a7dbbc43649ecdaa754))
* quick fixes to README to match config file ([da6df54](https://github.com/biothings/biomedical_id_resolver.js/commit/da6df549d54837d340501a8424e7a7166be241a0))
* reformatting, add mappings + ID prefixes ([546f5a7](https://github.com/biothings/biomedical_id_resolver.js/commit/546f5a70f6f2a003e737e27a0133440829d8095a))
* remove ChemicalSubstance support ([d78a30f](https://github.com/biothings/biomedical_id_resolver.js/commit/d78a30f7fd254812a808ca2479d0a3c759d7b3ff))
* simple changes ([df2139c](https://github.com/biothings/biomedical_id_resolver.js/commit/df2139c70c3be0292fddf9e1a6bcfa78961874d6))
* try to add backwards compatibility for ChemicalSubstance ([987ab97](https://github.com/biothings/biomedical_id_resolver.js/commit/987ab97fb52fb9baa94f3bd425f19857c130065e))
* update README to reflect config/types file changes ([3792602](https://github.com/biothings/biomedical_id_resolver.js/commit/3792602fcc361b65f5551b13451f3fa7f4d2b1a1))
* update rhea tests ([95ec48f](https://github.com/biothings/biomedical_id_resolver.js/commit/95ec48f23e9eca197f1d31db61815afb163fb845))
* update tests for ChemicalSubstance -> SmallMolecule ([0699af0](https://github.com/biothings/biomedical_id_resolver.js/commit/0699af00ede0dedbb063bf405c7d59159ce517f7))

### [3.9.1](https://github.com/biothings/biomedical_id_resolver.js/compare/v3.9.0...v3.9.1) (2021-06-22)

## [3.9.0](https://github.com/kevinxin90/biomedical_id_resolver.js/compare/v3.8.0...v3.9.0) (2021-04-27)


### Features

* :sparkles: support resolving RHEA chemical ids ([11a5aac](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/11a5aacde7f313eec61be61b42fbf68a0fa507d3))

## [3.8.0](https://github.com/kevinxin90/biomedical_id_resolver.js/compare/v3.7.0...v3.8.0) (2021-04-27)


### Bug Fixes

* :bug: change NCBIGENE to NCBIGene ([267548e](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/267548e507dfa6d93eaa12b831eb195f6a1866c9))


### Refactor

* :recycle: adjust to new syntax in biolink model package ([b7d53ab](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/b7d53ab4be5e8374e2188c00c3a374295e4a6844))

## [3.7.0](https://github.com/kevinxin90/biomedical_id_resolver.js/compare/v3.6.0...v3.7.0) (2021-04-23)


### Features

* :sparkles: use json request body for biothings api ([f8edcb8](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/f8edcb80c313994f81d9c9fbfa95492f22a31837))

## [3.6.0](https://github.com/kevinxin90/biomedical_id_resolver.js/compare/v3.5.0...v3.6.0) (2021-04-23)


### Features

* :sparkles: support biocarta as pathway id ([c8ac35e](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/c8ac35ee053c11a8f5c42ba52bc65755547e743b))

## [3.5.0](https://github.com/kevinxin90/biomedical_id_resolver.js/compare/v3.4.3...v3.5.0) (2021-03-26)


### Features

* :sparkles: support resolving protein ids ([fc5c899](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/fc5c8996552da7a0a73a965a8ab5b1640a7b19a1))

### [3.4.3](https://github.com/kevinxin90/biomedical_id_resolver.js/compare/v3.4.2...v3.4.3) (2021-03-25)


### Bug Fixes

* :bug: if comma in input, classify as irresolvable ([df92f2a](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/df92f2a04c2764f46714ff649d642637bd094630))

### [3.4.2](https://github.com/kevinxin90/biomedical_id_resolver.js/compare/v3.4.1...v3.4.2) (2021-03-25)


### Features

* :sparkles: support translating LINCS id ([ec2f8f4](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/ec2f8f4ac7175a5722e99e93f0f85033fa23b1d7))


### Refactor

* :recycle: update the logics of generateDBID to handle input with multiple colons ([6bb755f](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/6bb755f6bc406e894f749bf8b07d4e3625339d24))

### [3.4.1](https://github.com/kevinxin90/biomedical_id_resolver.js/compare/v3.4.0...v3.4.1) (2021-03-24)


### Bug Fixes

* :bug: fix wrong react id mapping ([78e27bd](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/78e27bd43b358435bf02c2f83abd11ecb2330e1b))

## [3.4.0](https://github.com/kevinxin90/biomedical_id_resolver.js/compare/v3.3.0...v3.4.0) (2021-03-18)


### Features

* :sparkles: add num_of_participants as property for pathwayy ([15eb081](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/15eb081e5d6e8e3daf75a82f64dc2b5c5f61f878))

## [3.3.0](https://github.com/kevinxin90/biomedical_id_resolver.js/compare/v3.2.1...v3.3.0) (2021-03-18)


### Features

* :sparkles: add additional attributes property for bioentity ([490ed46](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/490ed4653106c9ef3273fed963a11067bce6d2fa))
* :sparkles: handling cases when one query returns multiple hits from biothings api ([cdb66cb](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/cdb66cb58c4bda97f1193d83d081b042f817cab2))


### Bug Fixes

* :bug: fix bug that symbol not resolved to human gene by default ([d5eec4f](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/d5eec4f2862c84c047d1be9a6ffaef03c97fa124))


### Refactor

* :recycle: add additional attributes mapping to config ([56326b5](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/56326b58493e863291faa92d4fecf9dca3f23630))
* :recycle: add omim mapping in config ([2a59874](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/2a59874f123f02151f035c9da97fa231f3707805))

### [3.2.1](https://github.com/kevinxin90/biomedical_id_resolver.js/compare/v3.2.0...v3.2.1) (2021-03-12)


### Bug Fixes

* :bug: fix wrong uniprot id ([79be269](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/79be269e2f4480cdce62a4d21cf7e0283cf552e2))

## [3.2.0](https://github.com/kevinxin90/biomedical_id_resolver.js/compare/v3.1.1...v3.2.0) (2021-03-04)


### Features

* :sparkles: include additional mydisease.info fields for UMLS ID ([454fc00](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/454fc0084e80edf5b38de3895c9d19bf5f947992))
* :sparkles: support resolving Disease GARD ID ([af856d4](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/af856d44fceda1b7192058a7cd0eb16a3aaf445b))

### [3.1.1](https://github.com/kevinxin90/biomedical_id_resolver.js/compare/v3.1.0...v3.1.1) (2021-03-02)


### Bug Fixes

* :bug: fix wrong export method ([f605062](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/f605062a1d63ae1dc117bb25c01df2f91d11062f))

## [3.1.0](https://github.com/kevinxin90/biomedical_id_resolver.js/compare/v3.0.0...v3.1.0) (2021-03-02)


### Features

* :sparkles: add function to generate invalid responses ([94ea086](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/94ea0867e8a0b7392cb4df3e889ae68c8dae4a7a))

## [3.0.0](https://github.com/kevinxin90/biomedical_id_resolver.js/compare/v2.3.1...v3.0.0) (2021-03-01)


### Features

* :sparkles: Add new Validator which traverse biolink model ([4efb49e](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/4efb49e3a27e8cd11c2aa7fd02dab4dab61dd4d3))
* :sparkles: add semanticType as a property for BioEntity class ([c549e90](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/c549e908010dd4c4b87aba02d8eb64cce62675d6))
* :sparkles: create resolver that can traverse biolink hierarchy ([ab3de10](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/ab3de101818babeb75bc22ed5ff474ffc456f4bc))


### Bug Fixes

* :bug: fix mismatch with official biolink model ([30685df](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/30685dfc6f2640fbe6daedb1353c5a71fed971b2))


### Refactor

* :fire: remove DiseaseOrPhenotypicFeature from config ([713ec27](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/713ec27327e39289230c84979114b13e7cb29eaf))
* :fire: remove method defined in parent class ([ab91819](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/ab9181952758f8bfac0553293fb2a191c806c906))
* :recycle: add base resolver abstract class ([26dfcea](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/26dfcea1ca6d1186dcc7de548ecc514161531932))
* :recycle: add base resolver class ([061b421](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/061b421d10573ec4afe761196e370be63e613ffc))
* :recycle: add biolink handler module ([fcc1281](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/fcc128151e6902265abbb346ef7b0d2bce7bc4ad))
* :recycle: add interfaces ([d3db4f2](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/d3db4f28ca9b8773b3ab07381c74262ccfd410ef))
* :recycle: add irresolvable bioentity class ([7e2941c](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/7e2941cf43b7129a5df659ed4d4d082d01542bb7))
* :recycle: add new interfaces ([959ad4a](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/959ad4a8143e9953d5ebf655778c0fc2f47abada))
* :recycle: add semanticTypes and leafSemanticType properties ([a31cc4e](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/a31cc4eb9eaff353aeee284b44fc1a9a13358bf1))
* :recycle: add semanticTypes and leafSemanticType property ([5aff050](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/5aff050b66f045fe52328de04c60e515e6886b0f))
* :recycle: add valid property for BioLinkBasedValidator class ([4a1b304](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/4a1b30410de12c732219641a4378f429d30a2146))
* :recycle: add ValidatorObject interface ([6b0e7c6](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/6b0e7c6d7c39126cfc2cc721916f3b15c5145b1f))
* :recycle: base class implements IBioentity interface ([fd64d30](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/fd64d30433c4f980e85b800ae83ce41349dc4acd))
* :recycle: better naming ([21f4ee5](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/21f4ee5c63ec69f747305a821a6ed07d1c9d9920))
* :recycle: change export method to default ([123da5a](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/123da5a042a749c4fab32b6c28b473f0c5d4c6f9))
* :recycle: change valid to resolvable ([be7f56e](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/be7f56e7521ef227fc92b780d5b1ec0caa00b8ca))
* :recycle: create an async query function ([1473995](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/14739955541432ad6cbc89f69b8378e5c5fb57fb))
* :recycle: create base validator class ([2a468a3](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/2a468a38ad1f5ebe9ed022d40ab28c2ed160d2ee))
* :recycle: create default resolver class ([16c4cde](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/16c4cde7e404f61bf5e8fb4dbf25f57f7b7b6f19))
* :recycle: fix wrong class name ([71e15fe](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/71e15fe14f2f357a636cd9a1178dd5edfa485458))
* :recycle: if input not found, return irresolvable bioentity object ([35f1f84](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/35f1f84d3dcd13b51e4888d8219eeb29e8de89f3))
* :recycle: let default validator inherit from base validator ([b20d925](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/b20d9255d06f8da55c6409a7872ec66eb6c51dd0))
* :recycle: naming change ([14c0a82](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/14c0a82d7f2f7c33b9d4fb1b2574a1533f70cfc8))
* :recycle: remove method defined in inherited class ([b6e3c7e](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/b6e3c7e45a0820faa6824ab5447a7da572f01dc5))
* :recycle: restructure test folder ([39d14bf](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/39d14bfdf50fefa5f7d07a8e2780041fe1652d22))
* :recycle: update class and property name ([379a51d](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/379a51da8d550a367dd7ae7b4492e71741bf7b2c))
* :recycle: update entry point ([dede608](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/dede6085082887f26c3ec4db65d92edea429615d))
* :recycle: update function output types ([751b670](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/751b670fe761e44d16c1fdd8901e60e35907bcab))
* :recycle: update typescript type file ([9ec7f40](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/9ec7f402d0ade19197bd89f793003756a3c03eda))

### [2.3.1](https://github.com/kevinxin90/biomedical_id_resolver.js/compare/v2.3.0...v2.3.1) (2021-02-22)


### Bug Fixes

* :bug: enforce all values of dbIDs as an array of strings ([4f1d54b](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/4f1d54b77c16a70f54c3e0cdceb2d5033b6995e0))

## [2.3.0](https://github.com/kevinxin90/biomedical_id_resolver.js/compare/v2.2.1...v2.3.0) (2021-02-18)


### Features

* :sparkles: add an option to generate invalid inputs for all user Input ([49d9a40](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/49d9a401860b7b3d9404abe90be640fa9362a7b9))
* :sparkles: support for input with undefined semantic type ([f92c87f](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/f92c87f986dd4492c2cf43fae296ebe1b4543a2d))


### Bug Fixes

* :bug: fix wrong field name for Gene OMIM ID ([cd02a44](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/cd02a44465253b764a410c9e3fbb8bf98ea67098))


### Refactor

* :recycle: add function to generate id to type dict based on config ([ddfd505](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/ddfd50511b2e2904440fed5b9ea45ef927388f50))
* :recycle: check if id already exists in result before add invalid bioentity ([4e94038](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/4e94038683f13c615bd05b46d2a7e37c52f04c52))

### [2.2.1](https://github.com/kevinxin90/biomedical_id_resolver.js/compare/v2.2.0...v2.2.1) (2021-02-14)

## [2.2.0](https://github.com/kevinxin90/biomedical_id_resolver.js/compare/v2.1.1...v2.2.0) (2021-02-14)


### Features

* :sparkles: return itself as dbid if it's invalid ([8041f03](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/8041f0392e0aacf742ca7c8fe69db2b5045b389c))

### [2.1.1](https://github.com/kevinxin90/biomedical_id_resolver.js/compare/v2.1.0...v2.1.1) (2021-02-13)


### Bug Fixes

* :bug: fix curie construction bug ([de8b7ae](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/de8b7aefa49808492fc4469fd735012499be6276))

## [2.1.0](https://github.com/kevinxin90/biomedical_id_resolver.js/compare/v2.0.2...v2.1.0) (2021-01-29)


### Features

* :sparkles: add debug if api query fails ([86e691f](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/86e691feb9331aaada09bedea885b0f30ef1493e))
* :sparkles: Have ValidBioEntity and InValidBioEntity derive from abstract class BioEntity ([352d2a1](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/352d2a15286bd9b343757685a78bcddce6b55a30))
* :sparkles: integrate npm debug module for debugging ([2b1cc72](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/2b1cc72ef48830948b72551eeb2ac58d37702a8c))


### Refactor

* :fire: remove empty constructor for IDResolver ([94c0718](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/94c07184cf2e0f2680bd42fbcea18c91c4a8f2aa))
* :fire: remove unused functions ([51b7605](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/51b7605cccbd8d4a68a11942186649ad7436b6e5))
* :recycle: fix linting error ([3696995](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/369699586f8756591886e8a79fdcee7912cddbb5))
* :recycle: fix typescript linting error ([f55ac5a](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/f55ac5aedc6a92b1e4d84186ad6eab0a4e262e50))
* :recycle: fix typescript linting errors ([22a54b4](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/22a54b432403119bcc54bcd506a7836b1bf7f0ac))
* :recycle: import debug the correct way in TypeScript ([1d25ac3](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/1d25ac3fd7ceadbf659aa01a7cd02e4350467a31))
* :recycle: import debug the correct way in TypeScript ([c50f763](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/c50f7630109d4f9e06d5b38ce407cbbb39564e8e))

### [2.0.2](https://github.com/kevinxin90/biomedical_id_resolver.js/compare/v2.0.1...v2.0.2) (2021-01-28)

### [2.0.1](https://github.com/kevinxin90/biomedical_id_resolver.js/compare/v2.0.0...v2.0.1) (2021-01-28)


### Bug Fixes

* :bug: point entry point to built folder ([0e56cf2](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/0e56cf2515972982f147afa224437fffa5920d79))

## [2.0.0](https://github.com/kevinxin90/biomedical_id_resolver.js/compare/v0.1.1...v2.0.0) (2021-01-28)


### Features

* :boom: improve oop design of the ID Resolver module ([31a60f5](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/31a60f5acef3c01d3098e388bbde5f4bc167415f))
* :sparkles: add validator feature ([22b03ee](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/22b03eeaf898f257f27d8a59de99499dc8855c9f))


### Bug Fixes

* :bug: add colon between prefix and colon in generateCuries output ([175f98f](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/175f98f91f645204f92ce55b41dc7f208c83d648))


### Refactor

* :fire: remove unused files ([305ddd1](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/305ddd1cd965a41c5ada875824598b6ecd708a43))
* :recycle: add invalidbioentityclass ([dbe402b](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/dbe402bd2b14c31a41321b792388a0b00810bbb5))
* :recycle: add scheduler class ([fa56046](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/fa56046e1fa34ae5a30001029164798c8b571371))
* :recycle: create BioThings API Class to build axios query and parse response ([1adadb1](https://github.com/kevinxin90/biomedical_id_resolver.js/commit/1adadb1d02a7d57fda340d074bc5d063fa042f47))
