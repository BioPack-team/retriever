import { MetaDataObject } from '../../common/types';
import { APIMETA } from '../../config';

export abstract class QueryBuilder {
  protected semanticType: string;
  protected curies: string[];

  constructor(semanticType: string, curies: string[]) {
    this.semanticType = semanticType;
    this.curies = curies;
  }

  getAPIMetaData(semanticType: string = this.semanticType) {
    return APIMETA[semanticType];
  }

  abstract buildQueries(metadata: MetaDataObject, prefix: string, inputs: string[]): void;

  abstract buildOneQuery(metadata: MetaDataObject, prefix: string, curies: string[]): void;
}
