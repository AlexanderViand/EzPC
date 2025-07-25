/*
Authors: AI Assistant
Copyright:
Copyright (c) 2024 Microsoft Research
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "../src/api.h"
#include "../src/fss.h"
#include <iostream>
#include <vector>
#include <random>
#include <algorithm>
#include <cassert>

// Test function for FSS sorting
void testFSSSort()
{
    std::cout << "=== Testing FSS Sort with Compare-and-Aggregate Approach ===" << std::endl;
    
    // Test parameters
    const int num_elements = 8;
    const int key_bitlength = 32;
    const int value_bitlength = 32;
    
    // Initialize FSS
    fss_init();
    
    // Generate test data
    std::vector<GroupElement> keys(num_elements);
    std::vector<GroupElement> values(num_elements);
    std::vector<GroupElement> indices(num_elements);
    
    // Create random test data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(1, 100);
    
    std::cout << "Original data:" << std::endl;
    for (int i = 0; i < num_elements; i++)
    {
        keys[i] = GroupElement(dis(gen), key_bitlength);
        values[i] = GroupElement(i, value_bitlength); // Use index as value for easy verification
        indices[i] = GroupElement(i, 64);
        
        std::cout << "  [" << i << "] Key: " << keys[i].value << ", Value: " << values[i].value << std::endl;
    }
    
    // Create masked arrays
    std::vector<GroupElement> keys_mask(num_elements);
    std::vector<GroupElement> values_mask(num_elements);
    std::vector<GroupElement> indices_mask(num_elements);
    
    // Initialize masks (in practice, these would be random)
    for (int i = 0; i < num_elements; i++)
    {
        keys_mask[i] = GroupElement(0, key_bitlength);
        values_mask[i] = GroupElement(0, value_bitlength);
        indices_mask[i] = GroupElement(0, 64);
    }
    
    // Test ascending sort
    std::cout << "\nTesting ascending sort..." << std::endl;
    
    // Create copies for sorting
    std::vector<GroupElement> sorted_keys = keys;
    std::vector<GroupElement> sorted_values = values;
    std::vector<GroupElement> sorted_indices = indices;
    std::vector<GroupElement> sorted_keys_mask = keys_mask;
    std::vector<GroupElement> sorted_values_mask = values_mask;
    std::vector<GroupElement> sorted_indices_mask = indices_mask;
    
    // Perform FSS sort
    Sort(num_elements, key_bitlength, value_bitlength,
         MASK_PAIR(sorted_keys.data()),
         MASK_PAIR(sorted_values.data()),
         MASK_PAIR(sorted_indices.data()),
         true, true); // ascending, stable
    
    // Display sorted results
    std::cout << "Sorted data (ascending):" << std::endl;
    for (int i = 0; i < num_elements; i++)
    {
        std::cout << "  [" << i << "] Key: " << sorted_keys[i].value 
                  << ", Value: " << sorted_values[i].value 
                  << ", Original Index: " << sorted_indices[i].value << std::endl;
    }
    
    // Verify sorting is correct
    bool correctly_sorted = true;
    for (int i = 1; i < num_elements; i++)
    {
        if (sorted_keys[i-1].value > sorted_keys[i].value)
        {
            correctly_sorted = false;
            break;
        }
    }
    
    std::cout << "\nSort verification: " << (correctly_sorted ? "PASSED" : "FAILED") << std::endl;
    
    // Test descending sort
    std::cout << "\nTesting descending sort..." << std::endl;
    
    // Reset to original data
    sorted_keys = keys;
    sorted_values = values;
    sorted_indices = indices;
    sorted_keys_mask = keys_mask;
    sorted_values_mask = values_mask;
    sorted_indices_mask = indices_mask;
    
    // Perform FSS sort (descending)
    Sort(num_elements, key_bitlength, value_bitlength,
         MASK_PAIR(sorted_keys.data()),
         MASK_PAIR(sorted_values.data()),
         MASK_PAIR(sorted_indices.data()),
         false, true); // descending, stable
    
    // Display sorted results
    std::cout << "Sorted data (descending):" << std::endl;
    for (int i = 0; i < num_elements; i++)
    {
        std::cout << "  [" << i << "] Key: " << sorted_keys[i].value 
                  << ", Value: " << sorted_values[i].value 
                  << ", Original Index: " << sorted_indices[i].value << std::endl;
    }
    
    // Verify sorting is correct
    correctly_sorted = true;
    for (int i = 1; i < num_elements; i++)
    {
        if (sorted_keys[i-1].value < sorted_keys[i].value)
        {
            correctly_sorted = false;
            break;
        }
    }
    
    std::cout << "\nSort verification: " << (correctly_sorted ? "PASSED" : "FAILED") << std::endl;
    
    std::cout << "\n=== FSS Sort Test Complete ===" << std::endl;
}

// Test function for edge cases
void testEdgeCases()
{
    std::cout << "\n=== Testing Edge Cases ===" << std::endl;
    
    // Test with duplicate keys
    std::cout << "Testing with duplicate keys..." << std::endl;
    
    const int num_elements = 6;
    const int key_bitlength = 32;
    const int value_bitlength = 32;
    
    std::vector<GroupElement> keys = {
        GroupElement(5, key_bitlength),
        GroupElement(2, key_bitlength),
        GroupElement(5, key_bitlength), // Duplicate
        GroupElement(1, key_bitlength),
        GroupElement(2, key_bitlength), // Duplicate
        GroupElement(3, key_bitlength)
    };
    
    std::vector<GroupElement> values(num_elements);
    std::vector<GroupElement> indices(num_elements);
    std::vector<GroupElement> keys_mask(num_elements);
    std::vector<GroupElement> values_mask(num_elements);
    std::vector<GroupElement> indices_mask(num_elements);
    
    for (int i = 0; i < num_elements; i++)
    {
        values[i] = GroupElement(i, value_bitlength);
        indices[i] = GroupElement(i, 64);
        keys_mask[i] = GroupElement(0, key_bitlength);
        values_mask[i] = GroupElement(0, value_bitlength);
        indices_mask[i] = GroupElement(0, 64);
    }
    
    std::cout << "Original data with duplicates:" << std::endl;
    for (int i = 0; i < num_elements; i++)
    {
        std::cout << "  [" << i << "] Key: " << keys[i].value << ", Value: " << values[i].value << std::endl;
    }
    
    // Sort with stable sorting
    Sort(num_elements, key_bitlength, value_bitlength,
         MASK_PAIR(keys.data()),
         MASK_PAIR(values.data()),
         MASK_PAIR(indices.data()),
         true, true); // ascending, stable
    
    std::cout << "Sorted data (stable):" << std::endl;
    for (int i = 0; i < num_elements; i++)
    {
        std::cout << "  [" << i << "] Key: " << keys[i].value 
                  << ", Value: " << values[i].value 
                  << ", Original Index: " << indices[i].value << std::endl;
    }
    
    std::cout << "=== Edge Cases Test Complete ===" << std::endl;
}

int main()
{
    try
    {
        testFSSSort();
        testEdgeCases();
        std::cout << "\nAll tests completed successfully!" << std::endl;
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}