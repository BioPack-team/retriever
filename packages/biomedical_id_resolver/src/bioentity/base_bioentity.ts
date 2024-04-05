import { DBIdsObject, IBioEntity } from '../common/types';

export abstract class BioEntity implements IBioEntity {
  abstract get semanticTypes(): string[];

  abstract get semanticType(): string;

  abstract set semanticTypes(semanticTypes: string[]);

  abstract get primaryID(): string;

  abstract get label(): string;

  abstract get curies(): string[];

  abstract get dbIDs(): DBIdsObject;

  abstract get attributes(): DBIdsObject;
}
