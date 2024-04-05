export interface BioThingsAPIQueryResponse {
  query: string;
  [fieldName: string]: any;
}

export interface GrpedBioThingsAPIQueryResponse {
  [query: string]: BioThingsAPIQueryResponse[];
}

export interface BioThingsAPIFailedQueryResponse extends BioThingsAPIQueryResponse {
  notfound: true;
}

export type ResolvableSemanticTypes =
  | 'Gene'
  | 'Transcript'
  | 'Protein'
  | 'SequenceVariant'
  | 'SmallMolecule'
  | 'Drug'
  | 'Disease'
  | 'DiseaseOrPhenotypicFeature'
  | 'PhenotypicFeature'
  | 'ClinicalFinding'
  | 'MolecularActivity'
  | 'BiologicalProcess'
  | 'CellularComponent'
  | 'Pathway'
  | 'AnatomicalEntity'
  | 'Cell';

export enum ResolvableTypes {
  Gene = 'Gene',
  Transcript = 'Transcript',
  Protein = 'Protein',
  SequenceVariant = 'SequenceVariant',
  SmallMolecule = 'SmallMolecule',
  Drug = 'Drug',
  Disease = 'Disease',
  DiseaseOrPhenotypicFeature = 'DiseaseOrPhenotypicFeature',
  PhenotypicFeature = 'PhenotypicFeature',
  ClinicalFinding = 'ClinicalFinding',
  MolecularActivity = 'MolecularActivity',
  BiologicalProcess = 'BiologicalProcess',
  CellularComponent = 'CellularComponent',
  Cell = 'Cell',
  Pathway = 'Pathway',
  AnatomicalEntity = 'AnatomicalEntity',
}

export interface DBIdsObject {
  [prefix: string]: string[];
}

export type DBIdsObjectWithResolvableTypes = {
  [key in ResolvableSemanticTypes]: string[];
};

export interface TmpGrpedIDObject {
  [prefix: string]: (string | number)[];
}

export interface TmpGrpedResultObject {
  [curie: string]: TmpGrpedIDObject;
}

export interface APIFieldMappingObject {
  [prefix: string]: string[];
}

export interface MetaDataObject {
  id_ranks: string[];
  semantic: string;
  api_name: string;
  url: string;
  mapping: APIFieldMappingObject;
  additional_attributes_mapping?: APIFieldMappingObject;
}

export type MetaDataItemsObject = {
  [key: string]: MetaDataObject;
};

export interface ObjectWithStringKeyAndArrayValues {
  [key: string]: string[];
}

export interface DBIdsObjects {
  [key: string]: DBIdsObject;
}

export interface GroupedDBIDsObjects {
  [key: string]: DBIdsObjects;
}

export interface Buckets {
  [key: number]: any[];
}

export interface PromiseAllSettledOutput {
  status: string;
  [key: string]: any;
}

import { BioLink } from 'biolink-model';

export interface BioLinkHandlerClass {
  classTree: BioLink['classTree'];
}

export interface BioLinkEntityObject {
  parent: string;
  children: string[];
  description: string;
  id_prefixes: string[];
  name: string;
  addChild(child: string): void;
}

export interface ValidatorObject {
  resolvable: DBIdsObject;
  irresolvable: DBIdsObject;
  validate(): void;
}

export interface BioLinkBasedValidatorObject extends ValidatorObject {
  valid: DBIdsObject;
}

export interface IBioEntity {
  semanticTypes: string[];
  semanticType: string;
  primaryID: string;
  label: string;
  curies: string[];
  dbIDs: DBIdsObject;
  attributes: DBIdsObject;
}

export interface IndividualResolverOutput {
  [semanticType: string]: IBioEntity;
}

export interface ResolverOutput {
  [curie: string]: IBioEntity[];
}

export interface IResolver {
  resolve(userInput: unknown): Promise<ResolverOutput>;
}

export interface IDOBject {
  identifier: string;
  label: string;
}
export interface SRIBioEntity {
  primaryID: string; // SRI-preferred curie for this entity
  equivalentIDs: string[]; // curies that also resolve to this entity
  label: string; // SRI-preffered entity label
  labelAliases: string[]; // other labels for this entity
  // "main" semantic types for this entity, first is always the SRI-preferred type
  primaryTypes: string[];
  semanticTypes: string[]; // all types for this entity, up the hierarchy
  attributes?: DBIdsObject; // attributes attached to this entity
}

export interface IdentifierObject {
  identifier: string;
  label?: string;
}

export interface SRIResponseEntity {
  id: IdentifierObject;
  equivalent_identifiers: IdentifierObject[];
  type: string[];
  information_content: number;
}

export interface SRIResponse {
  [curie: string]: SRIResponseEntity;
}

export interface SRIResolverOutput {
  [curie: string]: SRIBioEntity;
}

export interface ResolverInput {
  [type: string]: string[];
}
