"""Streaming tools related to the HDF5 format"""
from typing import Union, Dict, Optional, List, Iterable
from contextlib import AbstractContextManager
from dataclasses import field, dataclass
from pathlib import Path
import logging
import warnings

import numpy as np
import pandas as pd
from tables import File, Group, Table, Filters

from battdat.io.hdf import write_df_to_table
from battdat.schemas.column import ColumnSchema, RawData
from battdat.schemas import BatteryMetadata
from battdat import __version__

logger = logging.getLogger(__name__)


@dataclass
class HDF5Writer(AbstractContextManager):
    """Tool to write raw time series data to an HDF5 file incrementally

    Writes data to the ``raw_data`` key of a different dataset."""

    # Attributes defining where and how to write
    hdf5_output: Union[Path, str, File]
    """File or already-open HDF5 file in which to store data"""
    write_mode: str = 'a'
    """Mode to use when opening the HDF5 file. Ignored if :attr:`hdf5_output` is a ``File``."""
    metadata: BatteryMetadata = field(default_factory=BatteryMetadata)
    """Metadata describing the cell"""
    schema: ColumnSchema = field(default_factory=RawData)
    """Schema describing columns of the cell"""
    complevel: int = 0
    """Compression level. Can be between 0 (no compression) and 9 (maximum compression). Ignored if data table already exists"""
    complib: str = 'zlib'
    """Compression algorithm. Consult :func:`~pandas.read_hdf` for available options. Ignored if data table already exists"""
    key: str = ''
    """Name of an existing group in which to store the data"""
    buffer_size: int = 32768
    """Number of rows to collect in memory before writing to disk"""

    # State used only while in writing mode
    _file: Optional[File] = None
    """Handle to an open file"""
    _group: Optional[Group] = None
    """Group in which to store the data"""
    _dtype: Optional[np.dtype] = None
    """Dtype of records to be written"""
    _table: Optional[Table] = None
    """Pointer to the table being written"""
    _write_buffer: Optional[List[Dict]] = None
    """Index of the next step to be written"""

    def __enter__(self):
        self._write_buffer = list()
        owns_file = not isinstance(self.hdf5_output, File)
        entered = False

        try:
            # Open the store, if needed
            if owns_file:
                self._file = File(
                    self.hdf5_output,
                    mode=self.write_mode
                )
            else:
                self._file = self.hdf5_output

            key = self.key.strip('/')
            self._group = self._file.get_node('/' + key) if key else self._file.root

            # Write metadata to the file's root attributes
            root = self._file.root
            metadata_json = self.metadata.model_dump_json(exclude_none=True)
            if 'metadata' in root._v_attrs:
                existing_json = root._f_getattr('metadata')
                try:
                    metadata_differs = BatteryMetadata.model_validate_json(existing_json) != self.metadata
                except ValueError:
                    metadata_differs = True
                if metadata_differs:
                    warnings.warn('Metadata already in HDF5 differs from new metadata')
            root._v_attrs.metadata = metadata_json
            root._v_attrs.json_schema = self.metadata.model_json_schema()
            root._v_attrs.battdat_version = __version__

            # Get the table if it exists already
            if 'raw_data' in self._group:
                self._table = self._group['raw_data']
            entered = True
            return self
        finally:
            if not entered:
                try:
                    if owns_file and self._file is not None:
                        self._file.close()
                finally:
                    if not owns_file or self._file is None or not self._file.isopen:
                        self._table = self._group = self._file = self._write_buffer = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        owns_file = not isinstance(self.hdf5_output, File)
        try:
            if len(self._write_buffer) > 0:  # Ensure last rows are written
                self.flush()
        finally:
            try:
                if owns_file and self._file is not None:
                    self._file.close()
            finally:
                if not owns_file or self._file is None or not self._file.isopen:
                    self._table = self._group = self._file = self._write_buffer = None
        return False

    def write_row(self, row: Dict[str, Union[str, float, int]]) -> int:
        """Add a row to the data file

        Args:
            row: Row to be added to the HDF5 file
        Returns:
            Number of rows written to file. Writes only occur when a write buffer has filled
        """
        return self.write_rows((row,))

    def write_rows(self, rows: Iterable[Dict[str, Union[str, float, int]]]) -> int:
        """Add multiple rows to the data file

        Args:
            rows: Rows to be added to the HDF5 file
        Returns:
            Number of rows written to file. Writes only occur when a write buffer has filled
        """
        written = 0
        for row in rows:
            self._write_buffer.append(row.copy())
            if len(self._write_buffer) >= self.buffer_size:
                written += self.flush()
        return written

    def flush(self) -> int:
        """Write the current row buffer to the file

        Returns:
            Number of rows written
        """

        if self._table is None:
            # Make the table the first time
            filters = Filters(complevel=self.complevel, complib=self.complib)
            df = pd.DataFrame(self._write_buffer)
            self._table = write_df_to_table(self._file, self._group, name='raw_data', filters=filters, df=df)

            # Store the metadata
            self._table.attrs.metadata = self.schema.model_dump_json()
            self._table.attrs.json_schema = self.schema.model_json_schema()
        else:
            # Append rows to the "raw_data" key
            row = np.empty((1,), dtype=self._table.dtype)
            known_names = set(self._table.dtype.names)
            for new_row in self._write_buffer:
                if (new_keys := set(new_row.keys())) != known_names:
                    logger.warning(f'Row has different keys than the Table. New keys: ({", ".join(new_keys.difference(known_names))}.'
                                   f' Missing: {", ".join(known_names.difference(new_keys))}')
                for c in known_names:
                    row[c] = new_row[c]
                self._table.append(row)

        written = len(self._write_buffer)
        self._write_buffer.clear()
        return written
