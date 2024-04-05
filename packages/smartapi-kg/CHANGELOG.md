# Changelog

All notable changes to this project will be documented in this file. See [standard-version](https://github.com/conventional-changelog/standard-version) for commit guidelines.

## [4.0.0](https://github.com/biothings/smartapi-kg.js/compare/v3.9.0...v4.0.0) (2022-06-02)


### Features

* :sparkle: Merge pull request [#38](https://github.com/biothings/smartapi-kg.js/issues/38) from biothings/smartapi-id ([317e24d](https://github.com/biothings/smartapi-kg.js/commit/317e24d05753c9f5c528232ee9cbd77eec0ce655)), closes [#258](https://github.com/biothings/smartapi-kg.js/issues/258)
* read in x-translator infores-curie field ([366a2af](https://github.com/biothings/smartapi-kg.js/commit/366a2af5e7e097995e7893905c0f3d88fc3682ce))
* support annotated batchSize for ID chunking ([9a2952b](https://github.com/biothings/smartapi-kg.js/commit/9a2952b475647603262f5e1571d5a0dad1de3ef8))
* support for templating-related operation annotations ([186bb4f](https://github.com/biothings/smartapi-kg.js/commit/186bb4fff4021da9baf9fa8b51b35f50eeff3494))
* use smartapi ids instead of names for filtering, apiNames -> apiList ([b22f1a4](https://github.com/biothings/smartapi-kg.js/commit/b22f1a4c1a6d281e17e9d523d0a80ca1e88bf97a))


### Bug Fixes

* change infores-curie to infores which will be in smartapi yaml files ([dde5335](https://github.com/biothings/smartapi-kg.js/commit/dde5335cf8a8945288c44039659f260cdc004a55))
* support global missing APIs for warning log ([6666471](https://github.com/biothings/smartapi-kg.js/commit/6666471f66b720fd2f8f661ad1271e3132c9fd81))

## [3.9.0](https://github.com/biothings/smartapi-kg.js/compare/v3.8.1...v3.9.0) (2021-08-16)

### [3.8.1](https://github.com/kevinxin90/smartapi-kg.js/compare/v3.8.0...v3.8.1) (2021-05-22)


### Bug Fixes

* :bug: check "x-bte-kgs-operations" as an array ([ceb001f](https://github.com/kevinxin90/smartapi-kg.js/commit/ceb001fce23046ae0e6aabc999c450085c5815b0))
* :bug: fix issue when predicates is not found ([c504cd1](https://github.com/kevinxin90/smartapi-kg.js/commit/c504cd1c2c0fb36314fa1f3b1b6d784036f7010f))

## [3.8.0](https://github.com/kevinxin90/smartapi-kg.js/compare/v3.7.0...v3.8.0) (2021-05-10)

## [3.7.0](https://github.com/kevinxin90/smartapi-kg.js/compare/v3.6.0...v3.7.0) (2021-04-02)


### Features

* :sparkles: filter ops from /predicates if user pass in api list ([d0498b7](https://github.com/kevinxin90/smartapi-kg.js/commit/d0498b7435db66ab099e6df617d9667cde09b957))

## [3.6.0](https://github.com/kevinxin90/smartapi-kg.js/compare/v3.5.0...v3.6.0) (2021-04-01)


### Features

* :sparkles: support user only selecting a list of specs to load based on titles ([8670b6f](https://github.com/kevinxin90/smartapi-kg.js/commit/8670b6f7b370bc51fc37480dde50057c4ce69cae))
* :sparkles: support user providing a single smartapi spec ([43f54ce](https://github.com/kevinxin90/smartapi-kg.js/commit/43f54ce4333e2cd9d2ac50866cf82a3310373e79))

## [3.5.0](https://github.com/kevinxin90/smartapi-kg.js/compare/v3.4.0...v3.5.0) (2021-04-01)


### Features

* :sparkles: support smartapi spec without hits field ([7409353](https://github.com/kevinxin90/smartapi-kg.js/commit/7409353e2eb4090dca762bb9953bd60b5f1edf7c))

## [3.4.0](https://github.com/kevinxin90/smartapi-kg.js/compare/v3.3.0...v3.4.0) (2021-03-30)

## [3.3.0](https://github.com/kevinxin90/smartapi-kg.js/compare/v3.2.0...v3.3.0) (2021-03-30)


### Features

* :sparkles: support get ops from /predicates in sync mode ([b172162](https://github.com/kevinxin90/smartapi-kg.js/commit/b172162dca0c69d70749428d14dba1d78f46e932))

## [3.2.0](https://github.com/kevinxin90/smartapi-kg.js/compare/v3.1.1...v3.2.0) (2021-03-29)


### Features

* :sparkles: add query_operations field for TRAPI apis ([6b3cb5a](https://github.com/kevinxin90/smartapi-kg.js/commit/6b3cb5a3104cbbe0107ac4969931b03343f63c19))

### [3.1.1](https://github.com/kevinxin90/smartapi-kg.js/compare/v3.1.0...v3.1.1) (2021-03-11)


### Bug Fixes

* :bug: update tsconfig to include local copy of SmartAPI specs when compiling ([684d308](https://github.com/kevinxin90/smartapi-kg.js/commit/684d3082d3528b228792b885641582b65069130b))

## [3.1.0](https://github.com/kevinxin90/smartapi-kg.js/compare/v3.0.0...v3.1.0) (2021-03-11)

## [3.0.0](https://github.com/kevinxin90/smartapi-kg.js/compare/v2.2.12...v3.0.0) (2021-03-06)


### Features

* :sparkles: allow user to specify no parameters for constructMetaKGSync ([d075f65](https://github.com/kevinxin90/smartapi-kg.js/commit/d075f65a88b5d629a73e1a153764b9b62df3d5fb))
* :sparkles: remove biolink prefix for all ops returned from /predicates endpoint ([f83d8a5](https://github.com/kevinxin90/smartapi-kg.js/commit/f83d8a51c7cf6c936382c64f81b093327c3d064e))
* :sparkles: support sync and async loading ([e3c69b1](https://github.com/kevinxin90/smartapi-kg.js/commit/e3c69b17822a058f43256352af49ab485655072c))


### Bug Fixes

* :bug: axios call returning >200 status code count as error ([b493a1b](https://github.com/kevinxin90/smartapi-kg.js/commit/b493a1bfd8e1dcdb80e6c3441a4f3afb22bfe63c))

### [2.2.12](https://github.com/kevinxin90/smartapi-kg.js/compare/v2.2.11...v2.2.12) (2021-02-23)

### [2.2.11](https://github.com/kevinxin90/smartapi-kg.js/compare/v2.2.10...v2.2.11) (2021-02-23)

### [2.2.10](https://github.com/kevinxin90/smartapi-kg.js/compare/v2.2.6...v2.2.10) (2021-02-23)


### Bug Fixes

* :bug: update smartapi api query url to capture all specs ([f9abaae](https://github.com/kevinxin90/smartapi-kg.js/commit/f9abaaed2c327bd6ff34580ba0459984c2626a00))

### [2.2.6](https://github.com/kevinxin90/smartapi-kg.js/compare/v2.2.5...v2.2.6) (2021-02-09)

### [2.2.5](https://github.com/kevinxin90/smartapi-kg.js/compare/v2.2.3...v2.2.5) (2021-02-07)

### [2.2.3](https://github.com/kevinxin90/smartapi-kg.js/compare/v2.2.2...v2.2.3) (2021-01-27)

### [2.2.2](https://github.com/kevinxin90/smartapi-kg.js/compare/v2.2.1...v2.2.2) (2021-01-27)

### [2.2.1](https://github.com/kevinxin90/smartapi-kg.js/compare/v2.2.0...v2.2.1) (2021-01-27)
