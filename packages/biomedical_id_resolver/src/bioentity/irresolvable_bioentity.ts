import { DBIdsObject } from '../common/types';
import { getPrefixFromCurie, generateDBID } from '../utils';
import { BioEntity } from './base_bioentity';

export class IrresolvableBioEntity extends BioEntity {
  private _leafSemanticType: string;
  private _semanticTypes: string[];
  private curie: string;

  constructor(semanticType: string, curie: string) {
    super();
    this._leafSemanticType = semanticType;
    this.curie = curie;
  }

  get semanticTypes(): string[] {
    if (typeof this._semanticTypes === 'undefined') {
      return [this._leafSemanticType];
    }
    return this._semanticTypes;
  }

  get semanticType(): string {
    return this._leafSemanticType;
  }

  set semanticTypes(types: string[]) {
    this._semanticTypes = types;
  }

  get primaryID(): string {
    return this.curie;
  }

  get label(): string {
    return this.curie;
  }

  get curies(): string[] {
    return [this.curie];
  }

  get dbIDs(): DBIdsObject {
    return {
      [getPrefixFromCurie(this.curie)]: [this.curie],
    };
  }

  get attributes(): DBIdsObject {
    return {};
  }
}
