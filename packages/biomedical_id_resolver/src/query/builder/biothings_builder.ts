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
} from '../../common/types';
import { APIMETA, TIMEOUT, MAX_BIOTHINGS_INPUT_SIZE } from '../../config';
import {
  generateDBID,
  generateObjectWithNoDuplicateElementsInValue,
  appendArrayOrNonArrayObjectToArray,
  generateCurie,
} from '../../utils';
import { ResolvableBioEntity } from '../../bioentity/valid_bioentity';
import { IrresolvableBioEntity } from '../../bioentity/irresolvable_bioentity';
import { QueryBuilder } from './base_builder';
import Debug from 'debug';
const debug = Debug('biomedical-id-resolver:QueryBuilder');

export class BioThingsQueryBuilder extends QueryBuilder {
  static queryTemplate: string = 'q={inputs}&scopes={scopes}&fields={fields}&dotfield=true&species=human';

  private getReturnFields(fieldMapping: APIFieldMappingObject): string {
    return Object.values(fieldMapping).reduce((prev, current) => prev + current.join(',') + ',', '');
  }

  private getInputScopes(fieldMapping: APIFieldMappingObject, prefix: string): string {
    return fieldMapping[prefix].join(',');
  }

  private groupCuriesByPrefix(curies: string[]): ObjectWithStringKeyAndArrayValues {
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

  private getDBIDsHelper(records: BioThingsAPIQueryResponse[]): DBIdsObject {
    const res = {} as DBIdsObject;
    const mapping = APIMETA[this.semanticType].mapping;
    Object.keys(mapping).map((prefix) => {
      for (const fieldName of mapping[prefix]) {
        records.map((record) => {
          if (fieldName in record) {
            if (!(prefix in res)) {
              res[prefix] = [];
            }
            res[prefix] = appendArrayOrNonArrayObjectToArray(res[prefix], record[fieldName]);
          }
        });
      }
    });
    return generateObjectWithNoDuplicateElementsInValue(res);
  }

  private getAttributesHelper(records: BioThingsAPIQueryResponse[]): DBIdsObject {
    const res = {} as DBIdsObject;
    const mapping = APIMETA[this.semanticType].additional_attributes_mapping;
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
    return generateObjectWithNoDuplicateElementsInValue(res);
  }

  private groupResultByQuery(response: BioThingsAPIQueryResponse[]): GrpedBioThingsAPIQueryResponse {
    const result = {} as GrpedBioThingsAPIQueryResponse;
    for (const rec of response) {
      if (!(rec.query in result)) {
        result[rec.query] = [];
      }
      result[rec.query].push(rec);
    }
    return result;
  }

  getDBIDs(prefix: string, semanticType: string, response: BioThingsAPIQueryResponse[]): IndividualResolverOutput {
    const result = {};
    const grpedResponse = this.groupResultByQuery(response);
    for (const query in grpedResponse) {
      const curie = generateCurie(prefix, query);
      if (!('notfound' in grpedResponse[query][0])) {
        result[curie] = new ResolvableBioEntity(
          semanticType,
          this.getDBIDsHelper(grpedResponse[query]),
          this.getAttributesHelper(grpedResponse[query]),
        );
      } else {
        result[curie] = new IrresolvableBioEntity(semanticType, curie);
      }
    }
    return result;
  }

  buildOneQuery(metadata: MetaDataObject, prefix: string, inputs: string[]): Promise<IndividualResolverOutput> {
    const idReturnFields = this.getReturnFields(metadata.mapping);
    let attrReturnFields = '';
    if ('additional_attributes_mapping' in metadata) {
      attrReturnFields = this.getReturnFields(metadata.additional_attributes_mapping);
    }
    const returnFields = idReturnFields + attrReturnFields;
    const scopes = this.getInputScopes(metadata.mapping, prefix);
    const biothingsQuery = BioThingsQueryBuilder.queryTemplate
      .replace('{inputs}', inputs.join(','))
      .replace('{scopes}', scopes)
      .replace('{fields}', returnFields);
    debug(
      `One Axios Query is built--- method: post, url: ${metadata.url}, timeout: ${TIMEOUT}, data: ${biothingsQuery}, inputs: ${inputs}`,
    );
    const userAgent = `BTE/${process.env.NODE_ENV === 'production' ? 'prod' : 'dev'} Node/${process.version} ${
      process.platform
    }`;
    return axios({
      method: 'post',
      url: metadata.url,
      timeout: TIMEOUT,
      params: {
        fields: returnFields,
        dotfield: true,
        species: 'human',
      },
      // data: biothingsQuery,
      data: {
        q: inputs,
        scopes: scopes,
      },
      headers: {
        'content-type': 'application/json',
        'User-Agent': userAgent,
      },
    }).then((response) => this.getDBIDs(prefix, this.semanticType, response.data));
  }

  buildQueries(metadata: MetaDataObject, prefix: string, inputs: string[]) {
    return _.chunk(inputs, MAX_BIOTHINGS_INPUT_SIZE).map((batch) => this.buildOneQuery(metadata, prefix, batch));
  }

  build() {
    const grped = this.groupCuriesByPrefix(this.curies);
    return Object.keys(grped).reduce((prev, current) => {
      prev = [...prev, ...this.buildQueries(this.getAPIMetaData(this.semanticType), current, grped[current])];
      return prev;
    }, []);
  }
}
