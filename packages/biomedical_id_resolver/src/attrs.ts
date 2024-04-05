import axios from 'axios';
import _ from 'lodash';

import {
  MetaDataObject,
  APIFieldMappingObject,
  ObjectWithStringKeyAndArrayValues,
  DBIdsObject,
  BioThingsAPIQueryResponse,
  IndividualResolverOutput,
  GrpedBioThingsAPIQueryResponse,
} from './common/types';
import { APIMETA, TIMEOUT, MAX_BIOTHINGS_INPUT_SIZE } from './config';
import {
  generateDBID,
  generateObjectWithNoDuplicateElementsInValue,
  appendArrayOrNonArrayObjectToArray,
  generateCurie,
} from './utils';
import { BioLink } from 'biolink-model';

import Debug from 'debug';
const debug = Debug('biomedical-id-resolver:AddAttributes');

function groupCuriesByPrefix(curies: string[]): ObjectWithStringKeyAndArrayValues {
  const grped: ObjectWithStringKeyAndArrayValues = {};
  curies.map((curie) => {
    const prefix = curie.split(':')[0];
    if (!(prefix in grped)) {
      grped[prefix] = [];
    }
    grped[prefix].push(generateDBID(curie));
  });
  return generateObjectWithNoDuplicateElementsInValue(grped);
}

function getReturnFields(fieldMapping: APIFieldMappingObject): any {
  try {
    return Object.values(fieldMapping).reduce((prev, current) => prev + current.join(',') + ',', '');
  } catch (error) {
    return false;
  }
}

function getInputScopes(fieldMapping: APIFieldMappingObject, prefix: string): any {
  try {
    return fieldMapping[prefix].join(',');
  } catch (error) {
    return false;
  }
}

function groupResultByQuery(response: BioThingsAPIQueryResponse[]): GrpedBioThingsAPIQueryResponse {
  const result = {} as GrpedBioThingsAPIQueryResponse;
  for (const rec of response) {
    if (!(rec.query in result)) {
      result[rec.query] = [];
    }
    result[rec.query].push(rec);
  }
  return result;
}

function getAttributesHelper(records: BioThingsAPIQueryResponse[], semanticType: string, curie: string): any {
  const res = {} as DBIdsObject;
  const mapping = APIMETA[semanticType].additional_attributes_mapping;
  if (typeof mapping === 'undefined') {
    return res;
  }
  Object.keys(mapping).map((attr) => {
    for (const fieldName of mapping[attr]) {
      records.map((record) => {
        if (fieldName in record) {
          if (!(attr in res)) {
            res[attr] = [];
          }
          res[attr] = appendArrayOrNonArrayObjectToArray(res[attr], record[fieldName]);
        }
      });
    }
  });
  let final_res = {};
  final_res[curie] = generateObjectWithNoDuplicateElementsInValue(res);
  return final_res;
}

function getDBIDs(prefix: string, semanticType: string, response: BioThingsAPIQueryResponse[]): any {
  const grpedResponse = groupResultByQuery(response);
  let final_res = {};
  for (const query in grpedResponse) {
    const curie = generateCurie(prefix, query);
    if (!('notfound' in grpedResponse[query][0])) {
      let attributes = getAttributesHelper(grpedResponse[query], semanticType, curie);
      if (Object.keys(attributes).length) {
        final_res = {
          ...final_res,
          ...attributes,
        };
      }
    }
  }
  //combined attributes collected for this prefix IDs
  return final_res;
}

function buildOneQuery(
  metadata: MetaDataObject,
  prefix: string,
  inputs: string[],
  semanticType: string,
): Promise<IndividualResolverOutput> {
  const idReturnFields = getReturnFields(metadata.mapping);
  let attrReturnFields = '';
  if ('additional_attributes_mapping' in metadata) {
    attrReturnFields = getReturnFields(metadata.additional_attributes_mapping);
  }
  const returnFields = idReturnFields + attrReturnFields;
  const scopes = getInputScopes(metadata.mapping, prefix);
  debug(`inputs ${inputs}`);
  const userAgent = `BTE/${process.env.NODE_ENV === 'production' ? 'prod' : 'dev'} Node/${process.version} ${process.platform
    }`;
  return axios({
    method: 'post',
    url: metadata.url,
    timeout: TIMEOUT,
    params: {
      fields: returnFields,
      dotfield: true,
    },
    data: {
      q: inputs,
      scopes: scopes,
    },
    headers: {
      'content-type': 'application/json',
      'User-Agent': userAgent,
    },
  })
    .then((response) => getDBIDs(prefix, semanticType, response.data))
    .catch(() => undefined);
}

function buildQueries(
  metadata: MetaDataObject,
  prefix: string,
  inputs: string[],
  semanticType: string,
): Promise<IndividualResolverOutput>[] {
  if (inputs.length > MAX_BIOTHINGS_INPUT_SIZE) {
    return _.chunk(inputs, MAX_BIOTHINGS_INPUT_SIZE).map((batch) =>
      buildOneQuery(metadata, prefix, batch, semanticType),
    );
  } else {
    return [buildOneQuery(metadata, prefix, inputs, semanticType)];
  }
}

function getAPIMetaData(semanticType: string) {
  return APIMETA[semanticType];
}

function build(semanticType: string, curies: string[]): Promise<IndividualResolverOutput>[] {
  const grped = groupCuriesByPrefix(curies);
  return Object.keys(grped).reduce((prev: Promise<IndividualResolverOutput>[], current) => {
    prev = [...prev, ...buildQueries(getAPIMetaData(semanticType), current, grped[current], semanticType)];
    return prev;
  }, []);
}

function getSupportedType(category: string): string {
  let bl = new BioLink();
  bl.loadSync();
  const ancestors = bl.classTree.getAncestors(category).map((v) => v.name);
  let categories = [...ancestors, ...[category]];
  for (let i = 0; i < categories.length; i++) {
    const cat = categories[i];
    if (Object.hasOwnProperty.call(APIMETA, cat)) {
      return cat;
    }
  }
  return '';
}

export async function _getAttributes(idsByType: object): Promise<any> {
  debug(`Adding attributes of ${JSON.stringify(idsByType)}`);
  let promises: Promise<IndividualResolverOutput>[] = [];
  for (const type in idsByType) {
    const ids = idsByType[type];
    const supportedType = getSupportedType(type);
    if (ids) {
      if (supportedType) {
        debug(`Processing attributes of: ${JSON.stringify(type)}`);
        promises = [...promises, ...build(supportedType, ids)];
      } else {
        debug(`Cannot get attributes of type: ${JSON.stringify(type)}`);
      }
    }
  }
  let res: IndividualResolverOutput[] = await Promise.all(promises);
  let merged = {};
  res.forEach((res_obj) => {
    if (typeof res_obj === 'undefined') return;
    merged = {
      ...merged,
      ...res_obj,
    };
  });
  return merged;
}
