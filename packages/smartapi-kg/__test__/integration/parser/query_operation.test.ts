import exp from 'constants';
import QueryOperationObject from '../../../src/parser/query_operation';

describe("Test QueryOperationObject class", () => {
    describe("Test xBTEKGSOperation setter function", () => {
        test("missing fields should return undefined", () => {
            const op = {
                parameters: {
                    gene: "{inputs[0]}"
                },
                requestBody: {
                    id: "{inputs[1]"
                },
                supportBatch: false,
                inputs: [{
                    id: "NCBIGene",
                    semantic: "Gene"
                }],
                outputs: [{
                    id: "NCBIGene",
                    semantic: "Gene"
                }],
                predicate: "related_to",
                response_mapping: {},
                requestBodyType: "object",
                templateInputs: {},
                useTemplating: true,
            }
            const obj = new QueryOperationObject();
            obj.xBTEKGSOperation = op;
            expect(obj.inputSeparator).toBeUndefined();
            expect(obj.useTemplating).toBeTruthy();
            expect(Object.keys(obj.templateInputs)).toHaveLength(0);
            expect(obj.params).toHaveProperty('gene');
            expect(obj.request_body).toHaveProperty('id');
            expect(obj.requestBodyType).toBe('object');
            expect(obj.supportBatch).toBeFalsy();
            expect(obj.method).toBeUndefined();
            expect(obj.server).toBeUndefined();
        })
    })
})
