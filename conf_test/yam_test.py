import yaml
print(yaml.__version__)


if __name__ == '__main__':
    with open('conf_test/conf.yaml', 'r') as fin:
        data = yaml.load(fin)
    
    schema = data['tables'][0]['schema']
    print(schema)
    print(schema.split(','))
    for s in schema.split(','):
        field, field_type, *extra_params = s.split()
        field, field_type = field.strip(), field_type.strip().upper()
        if field_type == 'DATETIME':
            print(field, field_type, extra_params)
