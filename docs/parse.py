#!/usr/bin/env python3
import re
import os
from pathlib import Path
from typing import Dict, List, NamedTuple
from dataclasses import dataclass

@dataclass
class Parameter:
    name: str
    type_hint: str
    description: str
    default: str = None

@dataclass
class DocInfo:
    name: str
    summary: str
    parameters: List[Parameter]
    returns: str
    returns_description: str
    full_description: str

class RustDocParser:
    def __init__(self):
        self.classes = {}
        self.methods = {}
    
    def parse_numpy_docstring(self, docstring: str) -> DocInfo:
        """Parse NumPy-style docstring into structured data."""
        lines = [line.strip() for line in docstring.split('\n')]
        
        # Find summary (first non-empty line)
        summary = ""
        for line in lines:
            if line and not line.startswith('---'):
                summary = line
                break
        
        # Parse sections
        parameters = []
        returns = ""
        returns_description = ""
        full_description = []
        
        current_section = None
        current_param = None
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # Check for section headers
            if line in ['Parameters', 'Returns', 'Attributes']:
                current_section = line.lower()
                # Skip the dashes line
                if i + 1 < len(lines) and lines[i + 1].startswith('---'):
                    i += 1
                i += 1
                continue
            
            if current_section == 'parameters' or current_section == 'attributes':
                # Parameter line format: "param_name : type"
                param_match = re.match(r'^(\w+)\s*:\s*(.+)$', line)
                if param_match:
                    param_name = param_match.group(1)
                    param_type = param_match.group(2)
                    
                    # Get description from following lines
                    description_lines = []
                    j = i + 1
                    while j < len(lines) and lines[j] and not re.match(r'^\w+\s*:', lines[j]):
                        description_lines.append(lines[j])
                        j += 1
                    
                    parameters.append(Parameter(
                        name=param_name,
                        type_hint=param_type,
                        description=' '.join(description_lines)
                    ))
                    i = j - 1
            
            elif current_section == 'returns':
                if line and not line.startswith('---'):
                    if not returns:
                        returns = line
                    else:
                        returns_description += f" {line}"
            
            elif not current_section and line and line != summary:
                full_description.append(line)
            
            i += 1
        
        return DocInfo(
            name="",
            summary=summary,
            parameters=parameters,
            returns=returns,
            returns_description=returns_description.strip(),
            full_description=' '.join(full_description).strip()
        )
    
    def extract_class_info(self, content: str) -> Dict[str, DocInfo]:
        """Extract class information from Rust content."""
        # Pattern to match class with docstring
        pattern = r'((?:///.*?\n)+).*?#\[pyclass\].*?pub struct (\w+)'
        
        classes = {}
        for match in re.finditer(pattern, content, re.DOTALL):
            docstring_block = match.group(1)
            class_name = match.group(2)
                        
            # Extract docstring content
            doc_lines = re.findall(r'/// (.*)', docstring_block)
            docstring = '\n'.join(doc_lines).strip()
            
            # Skip if no meaningful docstring content
            if not docstring or len(doc_lines) <= 1:
                # The len(doc_lines) <= 1 is currently a hack as 
                # structs with no docstrings that share prefixes
                # with other structs in the same file inherit a
                # one line docstring from the prefix-shared struct
                continue
            
            doc_info = self.parse_numpy_docstring(docstring)
            doc_info.name = class_name
            classes[class_name] = doc_info
        
        return classes
    
    def extract_method_info(self, content: str) -> Dict[str, Dict[str, DocInfo]]:
        """Extract method information from Rust content, grouped by class."""
        # First, find all #[pymethods] blocks and their associated classes
        pymethods_pattern = r'#\[pymethods\]\s*impl\s+(\w+)\s*\{(.*?)\n\}'
        
        class_methods = {}
        
        for match in re.finditer(pymethods_pattern, content, re.DOTALL):
            class_name = match.group(1)
            methods_block = match.group(2)
            
            # Extract methods from this specific block
            method_pattern = r'((?:///.*?\n)+).*?(?:pub\s+)?fn\s+(\w+)\((.*?)\)(?:\s*->\s*(.*?))?(?:\s*\{|(?=\s*where))'
            
            methods = {}
            for method_match in re.finditer(method_pattern, methods_block, re.DOTALL):
                docstring_block = method_match.group(1)
                method_name = method_match.group(2)
                params_str = method_match.group(3)
                return_type = method_match.group(4) if method_match.group(4) else None
                
                # Extract docstring content
                doc_lines = re.findall(r'/// (.*)', docstring_block)
                docstring = '\n'.join(doc_lines).strip()
                
                # Skip methods without meaningful docstrings
                if not docstring:
                    continue
                
                doc_info = self.parse_numpy_docstring(docstring)
                doc_info.name = method_name
                
                # Parse Rust function parameters
                rust_params = self.parse_rust_function_params(params_str)
                
                # Store Rust parameters and return type for signature generation
                doc_info.rust_params = rust_params
                doc_info.rust_return_type = return_type
                
                # Parse any pyo3 signature defaults
                pyo3_defaults = self.parse_pyo3_signature(content, method_name)
                
                # Match with docstring parameters and add defaults
                for param in doc_info.parameters:
                    if param.name in pyo3_defaults:
                        param.default = pyo3_defaults[param.name]
                
                methods[method_name] = doc_info
            
            if methods:  # Only add if there are documented methods
                class_methods[class_name] = methods
        
        return class_methods
    
    def parse_rust_function_params(self, params_str: str) -> List[tuple]:
        """Parse Rust function parameters from function signature."""
        params = []
        if not params_str.strip():
            return params
        
        # Split parameters by comma, handling nested types
        param_parts = []
        current_param = ""
        paren_count = 0
        bracket_count = 0
        
        for char in params_str:
            if char == ',' and paren_count == 0 and bracket_count == 0:
                param_parts.append(current_param.strip())
                current_param = ""
            else:
                if char == '(':
                    paren_count += 1
                elif char == ')':
                    paren_count -= 1
                elif char == '[':
                    bracket_count += 1
                elif char == ']':
                    bracket_count -= 1
                current_param += char
        
        if current_param.strip():
            param_parts.append(current_param.strip())
        
        # Parse each parameter
        for param in param_parts:
            param = param.strip()
            if not param:
                continue
            
            # Handle &self, &mut self, self
            if param in ['&self', '&mut self', 'self']:
                params.append(('self', 'self'))
                continue
            
            # Split by colon to get name and type
            if ':' in param:
                name_part, type_part = param.split(':', 1)
                name = name_part.strip()
                rust_type = type_part.strip()
                
                # Convert common Rust types to Python types
                python_type = self.rust_type_to_python(rust_type)
                params.append((name, python_type))
        
        return params
    
    def rust_type_to_python(self, rust_type: str) -> str:
        """Convert Rust types to Python type hints."""
        # Remove references and mutability
        rust_type = re.sub(r'&(mut\s+)?', '', rust_type)
        
        type_mapping = {
            'f64': 'float',
            'f32': 'float',
            'i32': 'int',
            'i64': 'int',
            'u32': 'int',
            'u64': 'int',
            'usize': 'int',
            'bool': 'bool',
            'String': 'str',
            '&str': 'str',
            'Vec<[f64; 2]>': 'List[List[float]]',
            '[f64; 2]': 'List[float]',
        }
        
        # Handle Vec<T>
        vec_match = re.match(r'Vec<(.+)>', rust_type)
        if vec_match:
            inner_type = self.rust_type_to_python(vec_match.group(1))
            return f'List[{inner_type}]'
        
        # Handle Option<T>
        option_match = re.match(r'Option<(.+)>', rust_type)
        if option_match:
            inner_type = self.rust_type_to_python(option_match.group(1))
            return f'Optional[{inner_type}]'

        # Handle PyResult<T>
        option_match = re.match(r'PyResult<(.+)>', rust_type)
        if option_match:
            inner_type = self.rust_type_to_python(option_match.group(1))
            return inner_type

        return type_mapping.get(rust_type, rust_type)
    
    def parse_pyo3_signature(self, content: str, method_name: str) -> Dict[str, str]:
        """Parse pyo3 signature attribute for default values."""
        defaults = {}
        
        # Look for pyo3 signature before the method
        pattern = rf'#\[pyo3\(signature = \((.*?)\)\].*?pub fn {method_name}\('
        match = re.search(pattern, content, re.DOTALL)
        
        if match:
            sig_params = match.group(1)
            for param in sig_params.split(','):
                param = param.strip()
                if '=' in param:
                    name, default = param.split('=', 1)
                    defaults[name.strip()] = default.strip()
        
        return defaults
    
    def generate_method_signature(self, method_name: str, doc_info: DocInfo) -> str:
        """Generate Python method signature from parsed info."""
        if method_name == 'new':
            method_name = '__init__'
        
        params = []
        
        # Get parameters from Rust function signature
        if hasattr(doc_info, 'rust_params') and doc_info.rust_params:
            for param_name, param_type in doc_info.rust_params:
                if param_name == 'self':
                    params.append('self')
                else:
                    # Try to match with docstring parameter for better type info
                    doc_param = next((p for p in doc_info.parameters if p.name == param_name), None)
                    if doc_param:
                        # Use docstring type if available, otherwise use converted Rust type
                        type_hint = doc_param.type_hint if doc_param.type_hint else param_type
                        param_str = f"{param_name}: {type_hint}"
                        if doc_param.default:
                            param_str += f" = {doc_param.default}"
                        params.append(param_str)
                    else:
                        # Use Rust type conversion
                        params.append(f"{param_name}: {param_type}")
        else:
            # Fallback to docstring parameters only
            if method_name != '__init__':
                params.append('self')
            
            for param in doc_info.parameters:
                param_str = f"{param.name}: {param.type_hint}"
                if param.default:
                    param_str += f" = {param.default}"
                params.append(param_str)
        
        # Add return type if available
        signature = f"{method_name}({', '.join(params)})"
        if hasattr(doc_info, 'rust_return_type') and doc_info.rust_return_type:
            # Convert Rust return type to Python
            python_return_type = self.rust_type_to_python(doc_info.rust_return_type)
            # Check if docstring has return type info
            if doc_info.returns and doc_info.returns != python_return_type:
                python_return_type = doc_info.returns
            signature += f" -> {python_return_type}"
        elif doc_info.returns:
            signature += f" -> {doc_info.returns}"
        
        return signature
    
    def generate_markdown(self, rust_files: List[str], output_file: str):
        """Generate markdown documentation from Rust files."""
        markdown_content = ["# API\n"]
        
        for rust_file in rust_files:
            if not os.path.exists(rust_file):
                print(f"Warning: {rust_file} not found")
                continue
            
            with open(rust_file, 'r') as f:
                content = f.read()
            
            # Extract classes and methods
            classes = self.extract_class_info(content)
            class_methods = self.extract_method_info(content)  # Now returns Dict[str, Dict[str, DocInfo]]
            
            # Generate markdown for each class
            for class_name, class_info in classes.items():
                markdown_content.append(f"## {class_name}\n")
                
                if class_info.summary:
                    markdown_content.append(f"{class_info.summary}\n")
                
                if class_info.full_description:
                    markdown_content.append(f"{class_info.full_description}\n")
                
                # Add attributes section if any
                if class_info.parameters:
                    markdown_content.append("### Attributes\n")
                    for param in class_info.parameters:
                        markdown_content.append(f"- **{param.name}** (`{param.type_hint}`): {param.description}")
                    markdown_content.append("")
                
                # Add methods for this specific class
                methods = class_methods.get(class_name, {})
                
                for method_name, method_info in methods.items():
                    signature = self.generate_method_signature(method_name, method_info)
                    markdown_content.append(f"### `{signature}`\n")
                    
                    if method_info.summary:
                        markdown_content.append(f"{method_info.summary}\n")
                    
                    if method_info.full_description:
                        markdown_content.append(f"{method_info.full_description}\n")
                    
                    # Parameters
                    if method_info.parameters:
                        markdown_content.append("**Parameters:**\n")
                        for param in method_info.parameters:
                            markdown_content.append(f"- **{param.name}** (`{param.type_hint}`): {param.description}")
                        markdown_content.append("")
                    
                    # Returns
                    if method_info.returns:
                        markdown_content.append("**Returns:**\n")
                        markdown_content.append(f"- `{method_info.returns}`: {method_info.returns_description}")
                        markdown_content.append("")
        
        # Write to file
        with open(output_file, 'w') as f:
            f.write('\n'.join(markdown_content))
        
        print(f"Documentation generated: {output_file}")

def main():
    """Main function to generate documentation."""
    parser = RustDocParser()
    
    rust_files = [
        "src/point/mod.rs",
        "src/polygon/mod.rs",
    ]
    
    output_file = "docs/api.md"
    
    parser.generate_markdown(rust_files, output_file)

if __name__ == "__main__":
    main()
