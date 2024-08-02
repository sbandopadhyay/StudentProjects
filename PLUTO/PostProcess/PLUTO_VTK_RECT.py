#this will work for VTK output with rectilinear grid from pluto
import vtk
import numpy as np
import os

def read_vtk_file(file_name):
    reader = vtk.vtkDataSetReader()
    reader.SetFileName(file_name)
    reader.ReadAllScalarsOn()
    reader.Update()
    return reader.GetOutput()

def get_coordinate(grid, coord_type):
    if coord_type == 'X' and hasattr(grid, 'GetXCoordinates'):
        return np.array([grid.GetXCoordinates().GetValue(i) for i in range(grid.GetXCoordinates().GetNumberOfTuples())])
    elif coord_type == 'Y' and hasattr(grid, 'GetYCoordinates'):
        return np.array([grid.GetYCoordinates().GetValue(i) for i in range(grid.GetYCoordinates().GetNumberOfTuples())])
    elif coord_type == 'Z' and hasattr(grid, 'GetZCoordinates'):
        return np.array([grid.GetZCoordinates().GetValue(i) for i in range(grid.GetZCoordinates().GetNumberOfTuples())])
    else:
        return None

def vtk_to_numpy_manual(vtk_array):
    n_values = vtk_array.GetNumberOfTuples() * vtk_array.GetNumberOfComponents()
    values = [vtk_array.GetValue(i) for i in range(n_values)]
    return np.array(values)

def get_XCoordinate_pluto(file_name):
    output = read_vtk_file(file_name)
    if isinstance(output, vtk.vtkRectilinearGrid):
        return get_coordinate(output, 'X')
    else:
        raise TypeError("Unsupported VTK data type for X coordinate extraction: {}".format(type(output).__name__))

def get_YCoordinate_pluto(file_name):
    output = read_vtk_file(file_name)
    if isinstance(output, vtk.vtkRectilinearGrid):
        return get_coordinate(output, 'Y')
    else:
        raise TypeError("Unsupported VTK data type for Y coordinate extraction: {}".format(type(output).__name__))

def get_ZCoordinate_pluto(file_name):
    output = read_vtk_file(file_name)
    if isinstance(output, vtk.vtkRectilinearGrid):
        return get_coordinate(output, 'Z')
    else:
        raise TypeError("Unsupported VTK data type for Z coordinate extraction: {}".format(type(output).__name__))

def get_FieldNames_pluto(file_name):
    output = read_vtk_file(file_name)
    variable_names = []
    if isinstance(output, vtk.vtkRectilinearGrid):
        cell_data = output.GetCellData()
        if cell_data:
            num_arrays = cell_data.GetNumberOfArrays()
            for i in range(num_arrays):
                array_name = cell_data.GetArrayName(i)
                if array_name:
                    variable_names.append(array_name)
    else:
        raise TypeError("Unsupported VTK data type for field name extraction: {}".format(type(output).__name__))
    return variable_names

def get_FieldValue(file_name, field_name):
    output = read_vtk_file(file_name)
    if isinstance(output, vtk.vtkRectilinearGrid):
        cell_data = output.GetCellData()
        if cell_data:
            array_data = cell_data.GetArray(field_name)
            if array_data:
                dims = [max(output.GetDimensions()[i] - 1, 1) for i in range(3)]
                np_data = vtk_to_numpy_manual(array_data).reshape(dims, order='F')
                return np_data
    else:
        raise TypeError("Unsupported VTK data type for field value extraction: {}".format(type(output).__name__))
    return None

def read_vtk_out(folder):
    vtk_info = {}
    vtk_out_path = os.path.join(folder, "vtk.out")
    with open(vtk_out_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            index = int(parts[0])
            time = float(parts[1])
            dt = float(parts[2])
            vtk_info[index] = {'time': time, 'dt': dt}
    return vtk_info

def get_time_and_dt(index, vtk_info):
    if index in vtk_info:
        return vtk_info[index]['time'], vtk_info[index]['dt']
    else:
        raise ValueError(f"No information found for index: {index}")

def get_time_and_dt_from_file_name(folder, file_name):
    vtk_info = read_vtk_out(folder)
    index_str = file_name.split('.')[1]
    index = int(index_str)
    if index in vtk_info:
        return vtk_info[index]['time'], vtk_info[index]['dt']
    else:
        raise ValueError(f"No information found for data file: {file_name}")

def process_vtk_file_by_number(folder, number):
    vtk_info = read_vtk_out(folder)
    data_file_name = f"data.{number:04d}.vtk"
    file_path = os.path.join(folder, data_file_name)
    
    print(f"Processing file: {file_path}")
    
    # Get time and dt
    time, dt = get_time_and_dt(number, vtk_info)
    print(f"Time: {time}, dt: {dt}")
    
    # Get X, Y, Z coordinates
    x_coord = get_XCoordinate_pluto(file_path)
    y_coord = get_YCoordinate_pluto(file_path)
    z_coord = get_ZCoordinate_pluto(file_path)
    
    print("X Coordinate:", x_coord)
    print("Y Coordinate:", y_coord)
    print("Z Coordinate:", z_coord)
    
    # Get field names
    field_names = get_FieldNames_pluto(file_path)
    print("Field Names:", field_names)
    
    # Get field values for each variable
    for variable in field_names:
        field_value = get_FieldValue(file_path, variable)
        print(f"Field Value for {variable}:")
        print(field_value)
