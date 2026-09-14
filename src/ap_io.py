from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


# =============================================================================
# PUBLIC CONFIGURATION OBJECT
# =============================================================================

@dataclass(frozen=True)
class IngestionPaths:
    """
    Input/output paths used by the ingestion layer.

    Parameters
    ----------
    raw_dir : pathlib.Path
        Main directory containing the raw Excel/CSV files.
    bronze_dir : pathlib.Path
        Directory where standardized Bronze snapshots are stored.
    weekly_sales_dir : pathlib.Path or None
        Optional folder containing periodic 'Artículos x Departamentos Venta'
        reports. If None, only Excel files inside `raw_dir` are scanned.
    pdf_tickets_dir : pathlib.Path or None
        Optional folder containing ticket/invoice PDFs.
    """
    raw_dir: Path
    bronze_dir: Path
    weekly_sales_dir: Path | None = None
    pdf_tickets_dir: Path | None = None


# =============================================================================
# CONSTANTS
# =============================================================================

SUPPORTED_EXCEL_EXTENSIONS = {'.xls', '.xlsx'}
SUPPORTED_TABLE_EXTENSIONS = {'.xls', '.xlsx', '.csv', '.parquet'}

REPORT_DATE_PATTERN = re.compile(r'(\d{2}/\d{2}/\d{4})')
ARTICLE_SALES_SUMMARY_STEMS = {
    'total_articulos',
    'total_articles',
}

BRONZE_FILENAMES = {
    'articulos': 'articulos_raw.parquet',
    'departamentos': 'departamentos_raw.parquet',
    'eventos': 'eventos_raw.parquet',
    'facturas': 'facturas_raw.parquet',
    'festivos': 'festivos_raw.parquet',
    'menu': 'menu_raw.parquet',
    'meteo_diaria': 'meteo_diaria_raw.parquet',
    'meteo_horaria': 'meteo_horaria_raw.parquet',
    'reservas': 'reservas_raw.parquet',
    'tickets': 'tickets_raw.parquet',
    'tips': 'tips_raw.parquet',
    'total_articles': 'total_articles_raw.parquet',
    'ventas': 'ventas_raw.parquet',
}


# =============================================================================
# CONSOLE HELPERS
# =============================================================================

def _print_header(title: str, verbose: bool = True) -> None:
    if not verbose:
        return

    line = '=' * 88
    print(f'\n{line}')
    print(title)
    print(line)


def _print_message(
    message: str,
    level: str = 'INFO',
    verbose: bool = True,
) -> None:
    if verbose:
        print(f'[{level}] {message}')


# =============================================================================
# TEXT / COLUMN NORMALIZATION
# =============================================================================

def _normalize_text(value: Any) -> str:
    """
    Normalize a value for robust comparisons.

    The normalization:
    - converts to lowercase;
    - removes accents;
    - collapses repeated whitespace;
    - strips leading/trailing whitespace.
    """
    if pd.isna(value):
        return ''

    text = unicodedata.normalize('NFKD', str(value))
    text = ''.join(
        character
        for character in text
        if not unicodedata.combining(character)
    )
    text = re.sub(r'\s+', ' ', text)

    return text.strip().lower()


def _normalize_column_name(value: Any) -> str:
    """Convert a column name to lowercase snake_case."""
    text = _normalize_text(value)
    text = re.sub(r'[^a-z0-9]+', '_', text)

    return text.strip('_')


def _clean_string_series(series: pd.Series) -> pd.Series:
    """Convert a Series to stripped nullable strings."""
    return (
        series
        .astype('string')
        .str.strip()
        .replace({'': pd.NA})
    )


def _normalize_dataframe_columns(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Return a copy with normalized snake_case column names."""
    result = dataframe.copy()
    result.columns = [
        _normalize_column_name(column)
        for column in result.columns
    ]
    return result


# =============================================================================
# SAFE TYPE CONVERSION
# =============================================================================

def _parse_report_date(value: Any) -> pd.Timestamp:
    """Extract a DD/MM/YYYY date from a free-text report cell."""
    match = REPORT_DATE_PATTERN.search(str(value))

    if match is None:
        return pd.NaT

    return pd.Timestamp(
        datetime.strptime(
            match.group(1),
            '%d/%m/%Y',
        )
    )


def _parse_excel_date(value: Any) -> pd.Timestamp:
    """Convert a scalar Excel date value to pandas.Timestamp."""
    if pd.isna(value):
        return pd.NaT

    if isinstance(value, pd.Timestamp):
        return value.normalize()

    if isinstance(value, datetime):
        return pd.Timestamp(value).normalize()

    if isinstance(value, date):
        return pd.Timestamp(value)

    text = str(value).strip()

    for date_format in (
        '%d/%m/%Y',
        '%Y-%m-%d',
        '%Y-%m-%d %H:%M:%S',
        '%d-%m-%Y',
    ):
        try:
            return pd.Timestamp(
                datetime.strptime(
                    text,
                    date_format,
                )
            ).normalize()
        except ValueError:
            continue

    return pd.to_datetime(
        value,
        errors='coerce',
        dayfirst=True,
    )


def _to_numeric(
    series: pd.Series,
) -> pd.Series:
    """
    Robust numeric conversion.

    Handles ordinary numeric values plus common European decimal formatting.
    """
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(
            series,
            errors='coerce',
        )

    text = (
        series
        .astype('string')
        .str.replace('\u00a0', '', regex=False)
        .str.replace('€', '', regex=False)
        .str.strip()
    )

    # Only apply European decimal conversion to strings that actually contain
    # commas. This avoids destroying normal decimal dots.
    comma_mask = text.str.contains(',', regex=False, na=False)
    text.loc[comma_mask] = (
        text.loc[comma_mask]
        .str.replace('.', '', regex=False)
        .str.replace(',', '.', regex=False)
    )

    return pd.to_numeric(
        text,
        errors='coerce',
    )


def _combine_date_and_time(
    date_values: pd.Series,
    time_values: pd.Series,
) -> pd.Series:
    """Combine separate Excel date and time columns into one datetime."""
    dates = pd.to_datetime(
        date_values,
        errors='coerce',
    ).dt.normalize()

    time_text = (
        time_values
        .astype('string')
        .str.strip()
    )

    time_delta = pd.to_timedelta(
        time_text,
        errors='coerce',
    )

    return dates + time_delta


# =============================================================================
# RAW EXCEL READING
# =============================================================================

def _read_excel_raw(
    file_path: str | Path,
) -> pd.DataFrame:
    """
    Read Excel without assuming a header row.

    Old .xls files are read with python-calamine because this was the most
    robust approach for the legacy POS exports used in the original project.
    Modern .xlsx files are read with openpyxl.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f'File not found: {path}'
        )

    if path.suffix.lower() not in SUPPORTED_EXCEL_EXTENSIONS:
        raise ValueError(
            f'Unsupported Excel extension: {path.suffix}. '
            f'Expected one of {sorted(SUPPORTED_EXCEL_EXTENSIONS)}.'
        )

    if path.suffix.lower() == '.xls':
        try:
            return pd.read_excel(
                path,
                header=None,
                dtype=object,
                engine='calamine',
            )
        except ImportError as exc:
            raise ImportError(
                'Reading .xls files requires python-calamine. '
                'Install it with: python -m pip install python-calamine'
            ) from exc

    return pd.read_excel(
        path,
        header=None,
        dtype=object,
        engine='openpyxl',
    )


def _table_from_first_row(
    raw: pd.DataFrame,
) -> pd.DataFrame:
    """Build a table using the first row as normalized column names."""
    columns = [
        _normalize_column_name(value)
        for value in raw.iloc[0].tolist()
    ]

    data = raw.iloc[1:].copy()
    data.columns = columns

    return data.reset_index(drop=True)


# =============================================================================
# REPORT METADATA
# =============================================================================

def _extract_value_after_colon(
    value: Any,
) -> str | None:
    """Return the text located after the first colon."""
    text = str(value)

    if ':' not in text:
        return None

    result = text.split(
        ':',
        maxsplit=1,
    )[1].strip()

    return result or None


def _extract_report_metadata(
    raw: pd.DataFrame,
) -> dict[str, Any]:
    """
    Extract common metadata from POS reports.

    Metadata currently supported:
    - report_start
    - report_end
    - report_generated_on
    - terminal_start
    - terminal_end
    - turn
    """
    metadata: dict[str, Any] = {
        'report_start': pd.NaT,
        'report_end': pd.NaT,
        'report_generated_on': pd.NaT,
        'terminal_start': None,
        'terminal_end': None,
        'turn': None,
    }

    for value in raw.to_numpy().ravel():
        if pd.isna(value):
            continue

        normalized = _normalize_text(value)

        if normalized.startswith('terminal inicial'):
            metadata['terminal_start'] = (
                _extract_value_after_colon(value)
            )

        elif normalized.startswith('terminal final'):
            metadata['terminal_end'] = (
                _extract_value_after_colon(value)
            )

        elif normalized.startswith('fecha inicial'):
            metadata['report_start'] = (
                _parse_report_date(value)
            )

        elif normalized.startswith('fecha final'):
            metadata['report_end'] = (
                _parse_report_date(value)
            )

        elif re.match(
            r'^fecha\s*:',
            normalized,
        ):
            metadata['report_generated_on'] = (
                _parse_report_date(value)
            )

        elif normalized.startswith('turno'):
            metadata['turn'] = (
                _extract_value_after_colon(value)
            )

    return metadata


def _add_report_metadata(
    dataframe: pd.DataFrame,
    source_path: Path,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    """Prepend source/report metadata to a parsed DataFrame."""
    result = dataframe.copy()

    result.insert(
        0,
        'source_file',
        source_path.name,
    )

    result.insert(
        1,
        'report_start',
        metadata['report_start'],
    )

    result.insert(
        2,
        'report_end',
        metadata['report_end'],
    )

    result.insert(
        3,
        'report_generated_on',
        metadata['report_generated_on'],
    )

    result.insert(
        4,
        'terminal_start',
        metadata['terminal_start'],
    )

    result.insert(
        5,
        'terminal_end',
        metadata['terminal_end'],
    )

    result.insert(
        6,
        'turn',
        metadata['turn'],
    )

    return result


# =============================================================================
# EXCEL FILE-TYPE DETECTION
# =============================================================================

def _detect_file_type(
    raw: pd.DataFrame,
) -> str:
    """
    Detect the supported dataset type from the Excel contents.

    The signatures below intentionally preserve the column structures used by
    the original TFM POS exports.
    """
    preview = raw.iloc[:30]

    normalized_cells = {
        _normalize_text(value)
        for value in preview.to_numpy().ravel()
        if pd.notna(value)
    }

    if 'articulos x departamentos venta' in normalized_cells:
        return 'article_sales'

    if 'documentos con emision de comprobante' in normalized_cells:
        return 'tickets'

    if 'resumen propinas' in normalized_cells:
        return 'tips'

    first_row = {
        _normalize_column_name(value)
        for value in raw.iloc[0].tolist()
        if pd.notna(value)
    }

    reservation_columns = {
        'fecha',
        'hora',
        'estado',
        'turno',
        'personas',
    }

    article_columns = {
        'articulo',
        'descripcion',
        'descripcion_abreviada',
        'cod_departamento_venta',
    }

    department_columns = {
        'codigo',
        'descripcion',
        'descripcion_abreviada',
    }

    menu_columns = {
        'articulo',
        'descripcion',
    }

    if reservation_columns.issubset(first_row):
        return 'reservations'

    if article_columns.issubset(first_row):
        return 'articles'

    if department_columns.issubset(first_row):
        return 'departments'

    if first_row == menu_columns:
        return 'menu'

    raise ValueError(
        'The Excel structure is not recognized by the current ingestion layer.'
    )


# =============================================================================
# EXCEL READERS
# =============================================================================

def read_article_sales(
    file_path: str | Path,
    raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Read an 'Artículos x Departamentos Venta' report.

    One output row represents one article inside one department for the period
    covered by the report.

    Department headers, repeated column headers, subtotal rows, blank rows and
    final total rows are ignored.
    """
    path = Path(file_path)

    raw = (
        _read_excel_raw(path)
        if raw is None
        else raw
    )

    metadata = _extract_report_metadata(
        raw
    )

    rows: list[dict[str, Any]] = []

    department_code: Any = pd.NA
    department_name: Any = pd.NA

    for _, row in raw.iterrows():
        code = (
            row.iloc[0]
            if len(row) > 0
            else pd.NA
        )

        description = (
            row.iloc[2]
            if len(row) > 2
            else pd.NA
        )

        units = (
            row.iloc[4]
            if len(row) > 4
            else pd.NA
        )

        amount = (
            row.iloc[6]
            if len(row) > 6
            else pd.NA
        )

        numeric_code = pd.to_numeric(
            pd.Series([code]),
            errors='coerce',
        ).iloc[0]

        numeric_units = pd.to_numeric(
            pd.Series([units]),
            errors='coerce',
        ).iloc[0]

        numeric_amount = pd.to_numeric(
            pd.Series([amount]),
            errors='coerce',
        ).iloc[0]

        has_description = (
            pd.notna(description)
            and str(description).strip() != ''
        )

        is_department = (
            pd.notna(numeric_code)
            and has_description
            and pd.isna(numeric_units)
            and pd.isna(numeric_amount)
        )

        if is_department:
            department_code = int(
                numeric_code
            )
            department_name = str(
                description
            ).strip()
            continue

        is_article = (
            pd.notna(numeric_code)
            and has_description
            and pd.notna(numeric_units)
            and pd.notna(numeric_amount)
        )

        if not is_article:
            continue

        rows.append(
            {
                'department_code': department_code,
                'department_name': department_name,
                'article_code': int(numeric_code),
                'article_name': str(description).strip(),
                'units': float(numeric_units),
                'amount': float(numeric_amount),
            }
        )

    data = pd.DataFrame(
        rows
    )

    return _add_report_metadata(
        data,
        path,
        metadata,
    )


def read_articles(
    file_path: str | Path,
    raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Read and standardize the POS article master."""
    path = Path(file_path)

    raw = (
        _read_excel_raw(path)
        if raw is None
        else raw
    )

    data = _table_from_first_row(
        raw
    )

    column_mapping = {
        'articulo': 'article_code',
        'codigo': 'article_code',
        'cod_articulo': 'article_code',
        'codigo_articulo': 'article_code',
        'descripcion': 'article_name',
        'descripcion_abreviada': 'article_short_name',
        'cod_departamento_venta': 'department_code',
        'departamento': 'department_code',
        'codigo_departamento': 'department_code',
        'precio': 'price',
        'pvp': 'price',
    }

    data = data.rename(
        columns={
            column: column_mapping.get(
                column,
                column,
            )
            for column in data.columns
        }
    )

    if 'article_code' not in data.columns:
        raise ValueError(
            'The article file does not contain an article code column.'
        )

    data['article_code'] = pd.to_numeric(
        data['article_code'],
        errors='coerce',
    ).astype('Int64')

    if 'department_code' in data.columns:
        data['department_code'] = pd.to_numeric(
            data['department_code'],
            errors='coerce',
        ).astype('Int64')

    if 'article_name' in data.columns:
        data['article_name'] = _clean_string_series(
            data['article_name']
        )

    if 'price' in data.columns:
        data['price'] = _to_numeric(
            data['price']
        )

    data = data[
        data['article_code'].notna()
    ].copy()

    data.insert(
        0,
        'source_file',
        path.name,
    )

    return data.reset_index(
        drop=True
    )


def read_menu(
    file_path: str | Path,
    raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Read and standardize the menu/carta export."""
    path = Path(file_path)

    raw = (
        _read_excel_raw(path)
        if raw is None
        else raw
    )

    data = _table_from_first_row(
        raw
    )

    column_mapping = {
        'articulo': 'article_code',
        'codigo': 'article_code',
        'codigo_articulo': 'article_code',
        'descripcion': 'article_name',
        'departamento': 'department_code',
        'precio': 'price',
        'pvp': 'price',
    }

    data = data.rename(
        columns={
            column: column_mapping.get(
                column,
                column,
            )
            for column in data.columns
        }
    )

    if 'article_code' in data.columns:
        data['article_code'] = pd.to_numeric(
            data['article_code'],
            errors='coerce',
        ).astype('Int64')

    if 'department_code' in data.columns:
        data['department_code'] = pd.to_numeric(
            data['department_code'],
            errors='coerce',
        ).astype('Int64')

    if 'article_name' in data.columns:
        data['article_name'] = _clean_string_series(
            data['article_name']
        )

    if 'price' in data.columns:
        data['price'] = _to_numeric(
            data['price']
        )

    data.insert(
        0,
        'source_file',
        path.name,
    )

    return data.reset_index(
        drop=True
    )


def read_departments(
    file_path: str | Path,
    raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Read and standardize the department master."""
    path = Path(file_path)

    raw = (
        _read_excel_raw(path)
        if raw is None
        else raw
    )

    data = _table_from_first_row(
        raw
    )

    column_mapping = {
        'codigo': 'department_code',
        'codigo_departamento': 'department_code',
        'descripcion': 'department_name',
        'departamento': 'department_name',
        'nombre_corto': 'department_short_name',
        'descripcion_corta': 'department_short_name',
        'descripcion_abreviada': 'department_short_name',
    }

    data = data.rename(
        columns={
            column: column_mapping.get(
                column,
                column,
            )
            for column in data.columns
        }
    )

    required_columns = {
        'department_code',
        'department_name',
    }

    missing_columns = sorted(
        required_columns.difference(
            data.columns
        )
    )

    if missing_columns:
        raise ValueError(
            'The department file is missing required columns: '
            + ', '.join(
                missing_columns
            )
        )

    data['department_code'] = pd.to_numeric(
        data['department_code'],
        errors='coerce',
    ).astype('Int64')

    data['department_name'] = _clean_string_series(
        data['department_name']
    )

    if 'department_short_name' in data.columns:
        data['department_short_name'] = (
            _clean_string_series(
                data['department_short_name']
            )
        )

    data = data[
        data['department_code'].notna()
        & data['department_name'].notna()
    ].copy()

    data.insert(
        0,
        'source_file',
        path.name,
    )

    return data.reset_index(
        drop=True
    )


def read_tickets(
    file_path: str | Path,
    raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Read the ticket list exported by the POS system.

    One output row represents one ticket/document.
    """
    path = Path(file_path)

    raw = (
        _read_excel_raw(path)
        if raw is None
        else raw
    )

    metadata = _extract_report_metadata(
        raw
    )

    dates = (
        raw.iloc[:, 1]
        .map(
            _parse_excel_date
        )
    )

    document_total = pd.to_numeric(
        raw.iloc[:, 5],
        errors='coerce',
    )

    receipt_count = pd.to_numeric(
        raw.iloc[:, 7],
        errors='coerce',
    )

    mask = (
        dates.notna()
        & raw.iloc[:, 3].notna()
        & document_total.notna()
        & receipt_count.notna()
    )

    data = pd.DataFrame(
        {
            'date': dates.loc[mask],
            'document_id': (
                raw.loc[
                    mask,
                    raw.columns[3],
                ]
                .astype('string')
                .str.strip()
            ),
            'document_total': (
                document_total
                .loc[mask]
                .astype(float)
            ),
            'receipt_count': (
                receipt_count
                .loc[mask]
                .astype('Int64')
            ),
        }
    ).reset_index(
        drop=True
    )

    return _add_report_metadata(
        data,
        path,
        metadata,
    )


def read_tips(
    file_path: str | Path,
    raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Read the tip summary exported by the POS system."""
    path = Path(file_path)

    raw = (
        _read_excel_raw(path)
        if raw is None
        else raw
    )

    metadata = _extract_report_metadata(
        raw
    )

    document_id = (
        raw.iloc[:, 0]
        .astype('string')
        .str.strip()
    )

    document_amount = pd.to_numeric(
        raw.iloc[:, 2],
        errors='coerce',
    )

    tip = pd.to_numeric(
        raw.iloc[:, 4],
        errors='coerce',
    )

    document_total = pd.to_numeric(
        raw.iloc[:, 6],
        errors='coerce',
    )

    mask = (
        document_id.notna()
        & document_amount.notna()
        & tip.notna()
        & document_total.notna()
        & ~document_id.str.upper().str.startswith(
            'TOTAL',
            na=False,
        )
    )

    data = pd.DataFrame(
        {
            'document_id': document_id.loc[mask],
            'document_amount': (
                document_amount
                .loc[mask]
                .astype(float)
            ),
            'tip': (
                tip
                .loc[mask]
                .astype(float)
            ),
            'document_total': (
                document_total
                .loc[mask]
                .astype(float)
            ),
        }
    ).reset_index(
        drop=True
    )

    return _add_report_metadata(
        data,
        path,
        metadata,
    )


def read_reservations(
    file_path: str | Path,
    raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Read and standardize the reservation export.

    Reservation and creation date/time pairs are also combined into datetime
    columns.
    """
    path = Path(file_path)

    raw = (
        _read_excel_raw(path)
        if raw is None
        else raw
    )

    data = _table_from_first_row(
        raw
    )

    column_mapping = {
        'fecha': 'reservation_date',
        'hora': 'reservation_time',
        'estado': 'status',
        'turno': 'shift',
        'personas': 'people',
        'origen': 'origin',
        'prescriptor': 'referrer',
        'fecha_anadida': 'created_date',
        'hora_anadida': 'created_time',
        'restaurante': 'restaurant',
        'tipo': 'reservation_type',
        'mesa': 'table',
        'zona': 'zone',
        'anotado_por': 'entered_by',
        'grupo': 'group',
        'referencia': 'reference',
        'codigo_de_referencia': 'reference_code',
    }

    data = data.rename(
        columns=column_mapping,
    )

    required_columns = {
        'reservation_date',
        'reservation_time',
        'status',
        'shift',
        'people',
    }

    missing_columns = sorted(
        required_columns.difference(
            data.columns
        )
    )

    if missing_columns:
        raise ValueError(
            'The reservations file is missing required columns: '
            + ', '.join(
                missing_columns
            )
        )

    data['reservation_date'] = pd.to_datetime(
        data['reservation_date'],
        errors='coerce',
        dayfirst=True,
    ).dt.normalize()

    if 'created_date' in data.columns:
        data['created_date'] = pd.to_datetime(
            data['created_date'],
            errors='coerce',
            dayfirst=True,
        ).dt.normalize()

    data['people'] = pd.to_numeric(
        data['people'],
        errors='coerce',
    ).astype('Int64')

    data['reservation_datetime'] = (
        _combine_date_and_time(
            data['reservation_date'],
            data['reservation_time'],
        )
    )

    if {
        'created_date',
        'created_time',
    }.issubset(
        data.columns
    ):
        data['created_datetime'] = (
            _combine_date_and_time(
                data['created_date'],
                data['created_time'],
            )
        )

    text_columns = [
        'status',
        'shift',
        'origin',
        'referrer',
        'restaurant',
        'reservation_type',
        'table',
        'zone',
        'entered_by',
        'group',
        'reference',
        'reference_code',
    ]

    for column in text_columns:
        if column in data.columns:
            data[column] = (
                data[column]
                .astype('string')
                .str.strip()
            )

    data.insert(
        0,
        'source_file',
        path.name,
    )

    preferred_columns = [
        'source_file',
        'reservation_datetime',
        'created_datetime',
        'reservation_date',
        'reservation_time',
        'status',
        'shift',
        'people',
        'origin',
        'referrer',
        'created_date',
        'created_time',
        'restaurant',
        'reservation_type',
        'table',
        'zone',
        'entered_by',
        'group',
        'reference',
        'reference_code',
    ]

    ordered_columns = [
        column
        for column in preferred_columns
        if column in data.columns
    ]

    remaining_columns = [
        column
        for column in data.columns
        if column not in ordered_columns
    ]

    return data[
        ordered_columns
        + remaining_columns
    ].reset_index(
        drop=True
    )


# =============================================================================
# GENERIC EXCEL DISPATCH
# =============================================================================

def read_excel_data_file(
    file_path: str | Path,
) -> tuple[str, pd.DataFrame]:
    """
    Detect and read one supported Excel file.

    Returns
    -------
    tuple[str, pandas.DataFrame]
        Dataset name and standardized DataFrame.
    """
    path = Path(
        file_path
    )

    raw = _read_excel_raw(
        path
    )

    file_type = _detect_file_type(
        raw
    )

    if file_type == 'article_sales':
        normalized_stem = _normalize_column_name(
            path.stem
        )

        dataset_name = (
            'total_articles'
            if normalized_stem
            in ARTICLE_SALES_SUMMARY_STEMS
            else 'ventas'
        )

        return (
            dataset_name,
            read_article_sales(
                path,
                raw=raw,
            ),
        )

    if file_type == 'articles':
        return (
            'articulos',
            read_articles(
                path,
                raw=raw,
            ),
        )

    if file_type == 'menu':
        return (
            'menu',
            read_menu(
                path,
                raw=raw,
            ),
        )

    if file_type == 'departments':
        return (
            'departamentos',
            read_departments(
                path,
                raw=raw,
            ),
        )

    if file_type == 'tickets':
        return (
            'tickets',
            read_tickets(
                path,
                raw=raw,
            ),
        )

    if file_type == 'tips':
        return (
            'tips',
            read_tips(
                path,
                raw=raw,
            ),
        )

    if file_type == 'reservations':
        return (
            'reservas',
            read_reservations(
                path,
                raw=raw,
            ),
        )

    raise ValueError(
        f'No reader implemented for file type: {file_type}'
    )


# =============================================================================
# CSV READING
# =============================================================================

def _read_csv_flexible(
    file_path: str | Path,
) -> pd.DataFrame:
    """
    Read a CSV using common separators.

    The first parsing that produces more than one column is retained.
    """
    path = Path(
        file_path
    )

    attempts = [
        {'sep': ','},
        {'sep': ';'},
        {'sep': '\t'},
    ]

    errors = []

    for kwargs in attempts:
        try:
            data = pd.read_csv(
                path,
                **kwargs,
            )

            if data.shape[1] > 1:
                return _normalize_dataframe_columns(
                    data
                )

        except Exception as exc:
            errors.append(
                str(exc)
            )

    raise ValueError(
        f'Could not parse CSV file {path.name}. '
        f'Parsing errors: {errors}'
    )


def _detect_csv_type(
    file_path: str | Path,
    dataframe: pd.DataFrame,
) -> str:
    """
    Detect Open-Meteo, holiday or event files.

    Current version uses both filename and available columns.
    """
    path = Path(
        file_path
    )

    name = _normalize_text(
        path.stem
    )

    columns = set(
        dataframe.columns
    )

    if (
        'open-meteo' in name
        or 'open_meteo' in name
        or 'meteo' in name
        or any(
            token in column
            for column in columns
            for token in [
                'temperature',
                'precipitation',
                'weather_code',
                'wind_speed',
            ]
        )
    ):
        return 'weather'

    if (
        'festiv' in name
        or 'holiday' in name
    ):
        return 'holidays'

    if (
        'evento' in name
        or 'event' in name
        or any(
            token in column
            for column in columns
            for token in [
                'evento',
                'event',
                'impacto',
                'intensidad',
            ]
        )
    ):
        return 'events'

    raise ValueError(
        'CSV type is not recognized by the current ingestion layer.'
    )



def _read_csv_blocks(
    file_path: str | Path,
) -> list[pd.DataFrame]:
    """
    Read one or more CSV tables separated by blank lines.

    This supports the Open-Meteo export used in this project, which contains:
    - one metadata table;
    - one hourly table;
    - one daily table.
    """
    path = Path(file_path)

    raw_text = path.read_text(
        encoding='utf-8-sig',
    )

    blocks: list[list[str]] = []
    current_block: list[str] = []

    for line in raw_text.splitlines():
        if line.strip():
            current_block.append(line)
        elif current_block:
            blocks.append(current_block)
            current_block = []

    if current_block:
        blocks.append(current_block)

    dataframes: list[pd.DataFrame] = []

    for block in blocks:
        if len(block) < 2:
            continue

        try:
            dataframe = pd.read_csv(
                StringIO(
                    '\n'.join(block)
                )
            )
        except Exception:
            continue

        if dataframe.shape[1] <= 1:
            continue

        dataframes.append(
            _normalize_dataframe_columns(
                dataframe
            )
        )

    return dataframes


def _standardize_weather_columns(
    dataframe: pd.DataFrame,
    frequency: str,
) -> pd.DataFrame:
    """Standardize the Open-Meteo column names used by the project."""
    data = dataframe.copy()

    mapping = {
        'temperature_2m_c': 'temperature_2m',
        'weather_code_wmo_code': 'weather_code',
        'rain_mm': 'rain_mm',
    }

    if frequency == 'daily':
        mapping.update(
            {
                'temperature_2m_mean_c': 'temperature_mean',
                'temperature_2m_max_c': 'temperature_max',
                'temperature_2m_min_c': 'temperature_min',
                'precipitation_sum_mm': 'precipitation_mm',
                'rain_sum_mm': 'rain_mm',
                'precipitation_hours_h': 'precipitation_hours',
                'wind_speed_10m_max_km_h': 'wind_speed_max_kmh',
                'sunshine_duration_s': 'sunshine_duration_s',
            }
        )

    return data.rename(
        columns={
            column: mapping.get(
                column,
                column,
            )
            for column in data.columns
        }
    )


def _weather_metadata(
    blocks: list[pd.DataFrame],
) -> dict[str, Any]:
    """Extract the metadata row from a combined Open-Meteo CSV."""
    for block in blocks:
        columns = set(
            block.columns
        )

        if {
            'latitude',
            'longitude',
        }.issubset(
            columns
        ) and 'time' not in columns:
            if block.empty:
                continue

            row = block.iloc[0]

            return {
                column: row[column]
                for column in [
                    'latitude',
                    'longitude',
                    'elevation',
                    'utc_offset_seconds',
                    'timezone',
                    'timezone_abbreviation',
                ]
                if column in block.columns
            }

    return {}


def _attach_weather_metadata(
    dataframe: pd.DataFrame,
    metadata: dict[str, Any],
) -> pd.DataFrame:
    """Attach constant location/timezone metadata to a weather table."""
    data = dataframe.copy()

    for column, value in metadata.items():
        if column not in data.columns:
            data[column] = value

    return data


def _daily_weather_from_hourly(
    hourly: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a daily weather table only when the source lacks a native daily block.
    """
    data = hourly.copy()

    if 'datetime' not in data.columns:
        return pd.DataFrame()

    data['date'] = (
        pd.to_datetime(
            data['datetime'],
            errors='coerce',
        )
        .dt.normalize()
    )

    numeric_columns = [
        column
        for column in data.select_dtypes(
            include='number'
        ).columns
        if column not in {
            'latitude',
            'longitude',
            'elevation',
            'utc_offset_seconds',
        }
    ]

    aggregation: dict[str, str] = {}

    for column in numeric_columns:
        normalized = column.lower()

        if (
            'rain' in normalized
            or 'precip' in normalized
        ):
            aggregation[column] = 'sum'

        elif (
            'wind' in normalized
            and (
                'max' in normalized
                or 'gust' in normalized
            )
        ):
            aggregation[column] = 'max'

        elif (
            'temperature' in normalized
            or 'temp' in normalized
        ):
            aggregation[column] = 'mean'

        elif (
            'sunshine' in normalized
            or 'duration' in normalized
        ):
            aggregation[column] = 'sum'

        else:
            aggregation[column] = 'mean'

    daily = (
        data.groupby(
            'date',
            as_index=False,
        )
        .agg(
            aggregation
        )
    )

    for column in [
        'latitude',
        'longitude',
        'elevation',
        'utc_offset_seconds',
        'timezone',
        'timezone_abbreviation',
    ]:
        if column in data.columns:
            non_missing = data[column].dropna()

            daily[column] = (
                non_missing.iloc[0]
                if not non_missing.empty
                else pd.NA
            )

    return daily


def read_weather_csv(
    file_path: str | Path,
) -> dict[str, pd.DataFrame]:
    """
    Read Open-Meteo weather data.

    The current project file contains three blocks in the same physical CSV:
    metadata, hourly observations and daily observations. This reader keeps the
    hourly and daily tables separate and prefers the native Open-Meteo daily
    table over a daily table reconstructed from hourly values.
    """
    path = Path(file_path)

    blocks = _read_csv_blocks(
        path
    )

    metadata = _weather_metadata(
        blocks
    )

    hourly: pd.DataFrame | None = None
    daily: pd.DataFrame | None = None

    for block in blocks:
        if 'time' not in block.columns:
            continue

        raw_time = (
            block['time']
            .astype('string')
        )

        has_explicit_clock = (
            raw_time
            .str.contains(
                r'T\d{1,2}:\d{2}',
                regex=True,
                na=False,
            )
            .any()
        )

        if has_explicit_clock:
            candidate = _standardize_weather_columns(
                block,
                frequency='hourly',
            )

            candidate = candidate.rename(
                columns={
                    'time': 'datetime',
                }
            )

            candidate['datetime'] = pd.to_datetime(
                candidate['datetime'],
                format='%Y-%m-%dT%H:%M',
                errors='coerce',
            )

            candidate = candidate.dropna(
                subset=[
                    'datetime',
                ]
            )

            hourly = _attach_weather_metadata(
                candidate,
                metadata,
            )

        else:
            candidate = _standardize_weather_columns(
                block,
                frequency='daily',
            )

            candidate = candidate.rename(
                columns={
                    'time': 'date',
                }
            )

            candidate['date'] = pd.to_datetime(
                candidate['date'],
                format='%Y-%m-%d',
                errors='coerce',
            ).dt.normalize()

            candidate = candidate.dropna(
                subset=[
                    'date',
                ]
            )

            if 'sunshine_duration_s' in candidate.columns:
                candidate['sunshine_hours'] = (
                    pd.to_numeric(
                        candidate[
                            'sunshine_duration_s'
                        ],
                        errors='coerce',
                    )
                    / 3600.0
                )

            daily = _attach_weather_metadata(
                candidate,
                metadata,
            )

    # Fallback: conventional one-table weather CSV.
    if hourly is None and daily is None:
        data = _read_csv_flexible(
            path
        )

        time_column = next(
            (
                column
                for column in [
                    'time',
                    'datetime',
                    'date_time',
                    'fecha_hora',
                ]
                if column in data.columns
            ),
            None,
        )

        date_column = next(
            (
                column
                for column in [
                    'date',
                    'fecha',
                ]
                if column in data.columns
            ),
            None,
        )

        if time_column is not None:
            parsed_time = pd.to_datetime(
                data[time_column],
                errors='coerce',
            )

            has_hour_information = (
                parsed_time.notna().any()
                and (
                    parsed_time.dt.hour.ne(0)
                    | parsed_time.dt.minute.ne(0)
                ).any()
            )

            if has_hour_information:
                hourly = _standardize_weather_columns(
                    data,
                    frequency='hourly',
                )

                hourly['datetime'] = (
                    parsed_time
                )

            else:
                daily = _standardize_weather_columns(
                    data,
                    frequency='daily',
                )

                daily['date'] = (
                    parsed_time
                    .dt.normalize()
                )

        elif date_column is not None:
            daily = _standardize_weather_columns(
                data,
                frequency='daily',
            )

            daily['date'] = pd.to_datetime(
                daily[date_column],
                errors='coerce',
                dayfirst=True,
            ).dt.normalize()

    result: dict[
        str,
        pd.DataFrame,
    ] = {}

    if hourly is not None and not hourly.empty:
        hourly = hourly.copy()

        if 'source_file' not in hourly.columns:
            hourly.insert(
                0,
                'source_file',
                path.name,
            )

        result[
            'meteo_horaria'
        ] = hourly.reset_index(
            drop=True
        )

    if daily is None and hourly is not None:
        daily = _daily_weather_from_hourly(
            hourly
        )

    if daily is not None and not daily.empty:
        daily = daily.copy()

        if 'source_file' not in daily.columns:
            daily.insert(
                0,
                'source_file',
                path.name,
            )

        result[
            'meteo_diaria'
        ] = daily.reset_index(
            drop=True
        )

    if not result:
        raise ValueError(
            'Weather CSV does not contain a recognizable hourly or daily '
            'Open-Meteo table.'
        )

    return result


def _parse_calendar_date_series(
    series: pd.Series,
) -> pd.Series:
    """
    Parse common calendar-date formats without confusing ISO dates.

    Passing `dayfirst=True` directly to pandas for an ISO value such as
    `2025-07-16` can misinterpret or reject the date. Known formats are parsed
    explicitly first; only unmatched values use a generic day-first fallback.
    """
    raw = (
        series.astype('string')
        .str.strip()
    )

    result = pd.Series(
        pd.NaT,
        index=series.index,
        dtype='datetime64[ns]',
    )

    formats = (
        (
            r'^\d{4}-\d{1,2}-\d{1,2}$',
            '%Y-%m-%d',
        ),
        (
            r'^\d{4}/\d{1,2}/\d{1,2}$',
            '%Y/%m/%d',
        ),
        (
            r'^\d{1,2}/\d{1,2}/\d{4}$',
            '%d/%m/%Y',
        ),
        (
            r'^\d{1,2}-\d{1,2}-\d{4}$',
            '%d-%m-%Y',
        ),
    )

    for pattern, date_format in formats:
        mask = (
            result.isna()
            & raw.str.match(
                pattern,
                na=False,
            )
        )

        if mask.any():
            result.loc[
                mask
            ] = pd.to_datetime(
                raw.loc[
                    mask
                ],
                format=date_format,
                errors='coerce',
            )

    remaining = (
        result.isna()
        & raw.notna()
        & raw.ne('')
    )

    if remaining.any():
        result.loc[
            remaining
        ] = pd.to_datetime(
            raw.loc[
                remaining
            ],
            errors='coerce',
            dayfirst=True,
        )

    return result.dt.normalize()


def read_holidays_csv(
    file_path: str | Path,
) -> pd.DataFrame:
    """
    Read and standardize a holiday calendar CSV.

    The Madrid/Pozuelo source currently uses ISO dates (`YYYY-MM-DD`), which
    are parsed explicitly to avoid accidental day/month inversion.

    Bronze holiday granularity:
        one holiday record per source row
    """
    path = Path(
        file_path
    )

    data = _read_csv_flexible(
        path
    )

    mapping = {
        'fecha': 'date',
        'festivo': 'holiday_name',
        'festivo_nombre': 'holiday_name',
        'nombre': 'holiday_name',
        'descripcion': 'holiday_name',
        'es_festivo': 'is_holiday',
        'dia_semana': 'day_name',
        'nivel': 'holiday_level',
    }

    data = data.rename(
        columns={
            column: mapping.get(
                column,
                column,
            )
            for column in data.columns
        }
    )

    if 'date' not in data.columns:
        raise ValueError(
            'Holiday CSV does not contain a date column.'
        )

    data['date'] = _parse_calendar_date_series(
        data['date']
    )

    if 'is_holiday' in data.columns:
        data['is_holiday'] = (
            pd.to_numeric(
                data['is_holiday'],
                errors='coerce',
            )
            .astype(
                'Int64'
            )
        )

    for column in [
        'holiday_name',
        'day_name',
        'holiday_level',
    ]:
        if column in data.columns:
            data[column] = (
                data[column]
                .astype(
                    'string'
                )
                .str.strip()
            )

    data.insert(
        0,
        'source_file',
        path.name,
    )

    return data


def read_events_csv(
    file_path: str | Path,
) -> pd.DataFrame:
    """
    Read and standardize the event calendar used by the project.

    The source uses `fecha_inicio` and `fecha_fin`. Each event is expanded to
    one row per affected calendar day so that it can later be integrated with a
    one-row-per-day Gold table.

    Bronze event granularity:
        event x affected calendar day

    The original event start/end dates and `event_id` are preserved.
    """
    path = Path(file_path)

    data = _read_csv_flexible(
        path
    )

    mapping = {
        'fecha': 'date',
        'fecha_inicio': 'event_start',
        'fecha_fin': 'event_end',
        'evento': 'event_name',
        'nombre_evento': 'event_name',
        'nombre': 'event_name',
        'categoria': 'event_category',
        'tipo': 'event_category',
        'subcategoria': 'event_subcategory',
        'ambito': 'event_scope',
        'ubicacion': 'event_location',
        'proximidad_la_roca': 'proximity_la_roca',
        'impacto_esperado': 'expected_impact',
        'impacto': 'expected_impact',
        'intensidad_sugerida': 'event_intensity',
        'intensidad': 'event_intensity',
        'direccion_demanda': 'demand_direction',
        'direccion': 'demand_direction',
        'franja_probable': 'probable_service_window',
        'segmento_cliente': 'customer_segment',
        'confianza_fecha': 'date_confidence',
        'confianza': 'date_confidence',
        'estimado': 'is_estimated',
        'feature_sugerida': 'suggested_feature',
        'fuente_tipo': 'source_type',
        'fuente': 'source_type',
        'notas_modelado': 'modeling_notes',
    }

    data = data.rename(
        columns={
            column: mapping.get(
                column,
                column,
            )
            for column in data.columns
        }
    )

    # Single-date event files remain supported.
    if (
        'date' in data.columns
        and 'event_start' not in data.columns
    ):
        data['date'] = pd.to_datetime(
            data['date'],
            errors='coerce',
            dayfirst=True,
        ).dt.normalize()

        data['event_start'] = (
            data['date']
        )

        data['event_end'] = (
            data['date']
        )

    if 'event_start' not in data.columns:
        raise ValueError(
            'Event CSV must contain either `fecha` or `fecha_inicio`.'
        )

    # The current event source uses ISO YYYY-MM-DD dates.
    data['event_start'] = pd.to_datetime(
        data['event_start'],
        format='%Y-%m-%d',
        errors='coerce',
    ).dt.normalize()

    if 'event_end' in data.columns:
        data['event_end'] = pd.to_datetime(
            data['event_end'],
            format='%Y-%m-%d',
            errors='coerce',
        ).dt.normalize()
    else:
        data['event_end'] = (
            data['event_start']
        )

    if 'event_intensity' in data.columns:
        data['event_intensity'] = pd.to_numeric(
            data['event_intensity'],
            errors='coerce',
        )

    if 'is_estimated' in data.columns:
        data['is_estimated'] = (
            pd.to_numeric(
                data['is_estimated'],
                errors='coerce',
            )
            .astype(
                'Int64'
            )
        )

    valid_range = (
        data['event_start'].notna()
        & data['event_end'].notna()
        & (
            data['event_end']
            >= data['event_start']
        )
    )

    data = data.loc[
        valid_range
    ].copy()

    data['event_duration_days'] = (
        data['event_end']
        - data['event_start']
    ).dt.days + 1

    expanded_rows: list[
        dict[str, Any]
    ] = []

    for _, row in data.iterrows():
        event_dates = pd.date_range(
            start=row['event_start'],
            end=row['event_end'],
            freq='D',
        )

        base = row.to_dict()

        for event_day_number, event_date in enumerate(
            event_dates,
            start=1,
        ):
            expanded = dict(
                base
            )

            expanded['date'] = (
                event_date.normalize()
            )

            expanded['event_day_number'] = (
                event_day_number
            )

            expanded_rows.append(
                expanded
            )

    expanded_data = pd.DataFrame(
        expanded_rows
    )

    expanded_data.insert(
        0,
        'source_file',
        path.name,
    )

    preferred_columns = [
        'source_file',
        'date',
        'event_id',
        'event_start',
        'event_end',
        'event_duration_days',
        'event_day_number',
        'event_name',
        'event_category',
        'event_subcategory',
        'event_scope',
        'event_location',
        'proximity_la_roca',
        'expected_impact',
        'event_intensity',
        'demand_direction',
        'probable_service_window',
        'customer_segment',
        'date_confidence',
        'is_estimated',
        'suggested_feature',
        'source_type',
        'source_url',
        'modeling_notes',
    ]

    ordered_columns = [
        column
        for column in preferred_columns
        if column in expanded_data.columns
    ]

    remaining_columns = [
        column
        for column in expanded_data.columns
        if column not in ordered_columns
    ]

    return expanded_data[
        ordered_columns
        + remaining_columns
    ].reset_index(
        drop=True
    )



def read_csv_data_file(
    file_path: str | Path,
) -> dict[str, pd.DataFrame]:
    """
    Detect and read one supported CSV.

    Open-Meteo is detected by filename before generic CSV parsing because the
    source file can contain several tables with different numbers of columns.
    """
    path = Path(
        file_path
    )

    normalized_name = (
        _normalize_text(
            path.stem
        )
        .replace(
            ' ',
            '_',
        )
    )

    if any(
        token in normalized_name
        for token in (
            'open-meteo',
            'open_meteo',
            'meteo',
            'weather',
        )
    ):
        return read_weather_csv(
            path
        )

    preview = _read_csv_flexible(
        path
    )

    file_type = _detect_csv_type(
        path,
        preview,
    )

    if file_type == 'weather':
        return read_weather_csv(
            path
        )

    if file_type == 'holidays':
        return {
            'festivos': read_holidays_csv(
                path
            )
        }

    if file_type == 'events':
        return {
            'eventos': read_events_csv(
                path
            )
        }

    raise ValueError(
        f'Unsupported CSV type: {file_type}'
    )


# =============================================================================
# PDF TICKET / INVOICE INGESTION
# =============================================================================

def _extract_pdf_text(
    file_path: str | Path,
) -> str:
    """Extract plain text from a PDF using PyMuPDF."""
    try:
        import pymupdf
    except ImportError as exc:
        raise ImportError(
            'PDF ingestion requires PyMuPDF. '
            'Install it with: python -m pip install pymupdf'
        ) from exc

    path = Path(
        file_path
    )

    document = pymupdf.open(
        path
    )

    extracted_text = '\n'.join(
        page.get_text(
            'text'
        )
        for page in document
    )

    document.close()

    return extracted_text


def _extract_pdf_words(
    file_path: str | Path,
) -> list[tuple]:
    """
    Extract positioned PDF words.

    Positioned words are important for restaurant receipts because PyMuPDF's
    plain-text reading order can place both payment labels before their values.
    Coordinates let us associate each amount with the correct printed row.
    """
    try:
        import pymupdf
    except ImportError as exc:
        raise ImportError(
            'PDF ingestion requires PyMuPDF. '
            'Install it with: python -m pip install pymupdf'
        ) from exc

    path = Path(
        file_path
    )

    document = pymupdf.open(
        path
    )

    words: list[tuple] = []

    for page_number, page in enumerate(
        document,
    ):
        for word in page.get_text(
            'words'
        ):
            words.append(
                (
                    *word,
                    page_number,
                )
            )

    document.close()

    return words


def _parse_decimal_token(
    value: Any,
) -> float:
    """
    Parse a Spanish/European decimal token safely.

    Examples
    --------
    '49,77' -> 49.77
    '1.234,56' -> 1234.56
    '1234.56' -> 1234.56
    """
    if value is None or pd.isna(
        value
    ):
        return np.nan

    token = str(
        value
    ).strip()

    token = re.sub(
        r'[^\d,.\-+]',
        '',
        token,
    )

    if not token:
        return np.nan

    if (
        ',' in token
        and '.' in token
    ):
        if token.rfind(
            ','
        ) > token.rfind(
            '.'
        ):
            token = (
                token.replace(
                    '.',
                    '',
                )
                .replace(
                    ',',
                    '.',
                )
            )
        else:
            token = token.replace(
                ',',
                '',
            )

    elif ',' in token:
        token = token.replace(
            ',',
            '.',
        )

    try:
        return float(
            token
        )
    except ValueError:
        return np.nan


def _extract_first_float(
    text: str,
    patterns: Iterable[str],
) -> float:
    """Extract the first decimal amount matching one of several regexes."""
    for pattern in patterns:
        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        if match:
            return _parse_decimal_token(
                match.group(1)
            )

    return np.nan


def _word_center_y(
    word: tuple,
) -> float:
    """Return the vertical center of one PyMuPDF word tuple."""
    return (
        float(
            word[1]
        )
        + float(
            word[3]
        )
    ) / 2.0


def _money_like_word(
    token: str,
) -> bool:
    """Return True for a standalone monetary-looking PDF token."""
    return bool(
        re.fullmatch(
            r'[+-]?\d{1,6}(?:[.,]\d{2})',
            str(
                token
            ).strip(),
        )
    )


def _find_label_word(
    words: list[tuple],
    label_pattern: str,
) -> tuple | None:
    """Find the first positioned word whose text matches a regex."""
    regex = re.compile(
        label_pattern,
        flags=re.IGNORECASE,
    )

    for word in words:
        if regex.fullmatch(
            str(
                word[4]
            ).strip()
        ):
            return word

    return None


def _extract_amount_on_label_line(
    words: list[tuple],
    label_pattern: str,
    y_tolerance: float = 3.0,
) -> tuple[float, bool]:
    """
    Extract the numeric amount printed on the same row as a label.

    Returns
    -------
    tuple
        (amount, label_found)

    Notes
    -----
    A present label with no amount on its row is interpreted later as a blank
    payment field, i.e. zero for that payment method. A missing label remains
    distinguishable from a printed blank.
    """
    label = _find_label_word(
        words,
        label_pattern,
    )

    if label is None:
        return (
            np.nan,
            False,
        )

    label_y = _word_center_y(
        label
    )

    candidates = [
        word
        for word in words
        if (
            int(
                word[-1]
            )
            == int(
                label[-1]
            )
            and float(
                word[0]
            )
            > float(
                label[2]
            )
            and abs(
                _word_center_y(
                    word
                )
                - label_y
            )
            <= y_tolerance
            and _money_like_word(
                word[4]
            )
        )
    ]

    if not candidates:
        return (
            np.nan,
            True,
        )

    nearest = min(
        candidates,
        key=lambda word: float(
            word[0]
        ),
    )

    return (
        _parse_decimal_token(
            nearest[4]
        ),
        True,
    )


def _extract_total_from_words(
    words: list[tuple],
) -> float:
    """Extract the receipt TOTAL amount from the printed total row."""
    total_words = [
        word
        for word in words
        if str(
            word[4]
        ).strip().upper() == 'TOTAL'
    ]

    # The final/lower TOTAL is the bill total. This also avoids the header
    # phrase "Total IVA".
    total_words = sorted(
        total_words,
        key=lambda word: (
            int(
                word[-1]
            ),
            float(
                word[1]
            ),
        ),
        reverse=True,
    )

    for label in total_words:
        label_y = _word_center_y(
            label
        )

        candidates = [
            word
            for word in words
            if (
                int(
                    word[-1]
                )
                == int(
                    label[-1]
                )
                and float(
                    word[0]
                )
                > float(
                    label[2]
                )
                and abs(
                    _word_center_y(
                        word
                    )
                    - label_y
                )
                <= 4.0
                and _money_like_word(
                    word[4]
                )
            )
        ]

        if candidates:
            rightmost = max(
                candidates,
                key=lambda word: float(
                    word[0]
                ),
            )

            return _parse_decimal_token(
                rightmost[4]
            )

    return np.nan


def _extract_tax_summary_from_words(
    words: list[tuple],
) -> tuple[
    float,
    float,
    float,
]:
    """
    Extract Base, VAT rate and VAT amount from the printed VAT table.

    The receipt layout is:

        Base       % IVA       Total IVA
        49,77      10,00       4,98

    Returns
    -------
    tuple
        (tax_base, vat_rate, vat_amount)
    """
    base_headers = [
        word
        for word in words
        if str(
            word[4]
        ).strip().lower() == 'base'
    ]

    for header in base_headers:
        header_page = int(
            header[-1]
        )

        header_y = _word_center_y(
            header
        )

        candidates = [
            word
            for word in words
            if (
                int(
                    word[-1]
                )
                == header_page
                and _money_like_word(
                    word[4]
                )
                and (
                    _word_center_y(
                        word
                    )
                    > header_y + 3.0
                )
                and (
                    _word_center_y(
                        word
                    )
                    < header_y + 30.0
                )
            )
        ]

        # The tax-detail row contains exactly three values, spatially ordered:
        # base, percentage and VAT amount.
        candidates = sorted(
            candidates,
            key=lambda word: float(
                word[0]
            ),
        )

        if len(
            candidates
        ) >= 3:
            values = [
                _parse_decimal_token(
                    word[4]
                )
                for word in candidates[:3]
            ]

            return (
                values[0],
                values[1],
                values[2],
            )

    return (
        np.nan,
        np.nan,
        np.nan,
    )


def _extract_pdf_identity_fields(
    text: str,
) -> dict[str, Any]:
    """Extract ticket identifier, date, time, table and server."""
    ticket_match = re.search(
        r'\b(\d{5,}TM\d+)\b',
        text,
        flags=re.IGNORECASE,
    )

    date_match = re.search(
        r'\b(\d{1,2}/\d{1,2}/\d{2,4})\b',
        text,
    )

    time_match = re.search(
        r'\b(\d{1,2}:\d{2}(?::\d{2})?)\b',
        text,
    )

    table_match = re.search(
        r'(?i)\bmesa\s*[:\-]?\s*([A-Za-z0-9_-]+)',
        text,
    )

    served_by_match = re.search(
        r'(?i)\ble\s+atendi[oó]\s*[:\-]?\s*([^\r\n]+)',
        text,
    )

    parsed_date = (
        pd.to_datetime(
            date_match.group(1),
            dayfirst=True,
            errors='coerce',
        ).normalize()
        if date_match
        else pd.NaT
    )

    return {
        'ticket_id': (
            ticket_match.group(1)
            if ticket_match
            else pd.NA
        ),
        'date': parsed_date,
        'time': (
            time_match.group(1)
            if time_match
            else pd.NA
        ),
        'table': (
            table_match.group(1)
            if table_match
            else pd.NA
        ),
        'served_by': (
            served_by_match.group(1).strip()
            if served_by_match
            else pd.NA
        ),
    }


def _resolve_payment_summary(
    total: float,
    cash_printed: float,
    cash_label_found: bool,
    card_printed: float,
    card_label_found: bool,
    tolerance: float = 0.02,
) -> dict[str, Any]:
    """
    Convert the printed payment section into analytically useful amounts.

    On these POS receipts, `EFECTIVO` can be the amount tendered by the customer,
    not necessarily the part of the bill paid in cash. Example: a EUR 60.50
    ticket may print EUR 65.00 in cash, implying EUR 4.50 change.

    We therefore preserve the printed cash amount as `cash_tendered`, derive
    `change_amount`, and store `cash_amount` as the cash actually applied to the
    bill. `cash_amount + card_amount` can then be reconciled against `total`.
    """
    if cash_label_found:
        cash_tendered = (
            0.0
            if pd.isna(
                cash_printed
            )
            else float(
                cash_printed
            )
        )
    else:
        cash_tendered = np.nan

    if card_label_found:
        card_amount = (
            0.0
            if pd.isna(
                card_printed
            )
            else float(
                card_printed
            )
        )
    else:
        card_amount = np.nan

    cash_amount = cash_tendered
    change_amount = np.nan

    if (
        pd.notna(
            total
        )
        and pd.notna(
            cash_tendered
        )
        and pd.notna(
            card_amount
        )
    ):
        printed_payment_total = (
            cash_tendered
            + card_amount
        )

        excess = (
            printed_payment_total
            - float(
                total
            )
        )

        if excess > tolerance:
            change_amount = excess
            cash_amount = max(
                cash_tendered
                - change_amount,
                0.0,
            )
        else:
            change_amount = 0.0

    cash_positive = (
        pd.notna(
            cash_amount
        )
        and cash_amount > tolerance
    )

    card_positive = (
        pd.notna(
            card_amount
        )
        and card_amount > tolerance
    )

    if (
        cash_positive
        and card_positive
    ):
        payment_method = 'mixed'
    elif cash_positive:
        payment_method = 'cash'
    elif card_positive:
        payment_method = 'card'
    elif (
        cash_label_found
        or card_label_found
    ):
        payment_method = 'none_or_zero'
    else:
        payment_method = 'unknown'

    return {
        'cash_tendered': cash_tendered,
        'cash_amount': cash_amount,
        'card_amount': card_amount,
        'change_amount': change_amount,
        'payment_method': payment_method,
    }


def read_pdf_ticket(
    file_path: str | Path,
) -> dict[str, Any]:
    """
    Parse the La Roca POS ticket/factura PDFs used by the current project.

    The parser combines plain text with positioned PDF words. Coordinates are
    necessary for the payment section because plain-text extraction can output:

        EFECTIVO:
        TARJETA:
        20,00
        34,75

    even though the values are visually printed on different rows.

    The raw extracted text is preserved for traceability.

    Returned monetary semantics
    ---------------------------
    base
        Taxable base printed in the VAT summary.
    vat_rate
        Printed VAT percentage.
    vat
        VAT amount (`Total IVA`), not the VAT percentage.
    total
        Final ticket total.
    cash_tendered
        Amount physically printed next to EFECTIVO.
    cash_amount
        Effective cash contribution after subtracting inferred change.
    card_amount
        Amount printed next to TARJETA.
    change_amount
        Positive excess tender inferred from the payment section.
    """
    path = Path(
        file_path
    )

    extracted_text = _extract_pdf_text(
        path
    )

    words = _extract_pdf_words(
        path
    )

    identity = _extract_pdf_identity_fields(
        extracted_text
    )

    if pd.isna(
        identity[
            'ticket_id'
        ]
    ):
        identity[
            'ticket_id'
        ] = path.stem

    total = _extract_total_from_words(
        words
    )

    if pd.isna(
        total
    ):
        total = _extract_first_float(
            extracted_text,
            patterns=[
                r'^\s*TOTAL\s*[:\-]?\s*([0-9]+(?:[.,][0-9]{2}))',
                r'\bTOTAL\b[^\d]{0,20}([0-9]+(?:[.,][0-9]{2}))',
            ],
        )

    (
        vat_base,
        vat_rate,
        vat_amount,
    ) = _extract_tax_summary_from_words(
        words
    )

    # Text fallback for PDF generators that do not preserve useful coordinates.
    if any(
        pd.isna(
            value
        )
        for value in [
            vat_base,
            vat_rate,
            vat_amount,
        ]
    ):
        tax_match = re.search(
            r'(?is)\bBase\s*%\s*IVA\s*Total\s*IVA\s*'
            r'([0-9]+(?:[.,][0-9]{2}))\s*'
            r'([0-9]+(?:[.,][0-9]{2}))\s*'
            r'([0-9]+(?:[.,][0-9]{2}))',
            extracted_text,
        )

        if tax_match:
            vat_base = _parse_decimal_token(
                tax_match.group(1)
            )
            vat_rate = _parse_decimal_token(
                tax_match.group(2)
            )
            vat_amount = _parse_decimal_token(
                tax_match.group(3)
            )

    (
        cash_printed,
        cash_label_found,
    ) = _extract_amount_on_label_line(
        words,
        label_pattern=r'EFECTIVO:?',
    )

    (
        card_printed,
        card_label_found,
    ) = _extract_amount_on_label_line(
        words,
        label_pattern=r'TARJETA:?',
    )

    payment = _resolve_payment_summary(
        total=total,
        cash_printed=cash_printed,
        cash_label_found=cash_label_found,
        card_printed=card_printed,
        card_label_found=card_label_found,
    )

    return {
        'source_file': path.name,
        'ticket_id': identity[
            'ticket_id'
        ],
        'date': identity[
            'date'
        ],
        'time': identity[
            'time'
        ],
        'table': identity[
            'table'
        ],
        'served_by': identity[
            'served_by'
        ],
        'base': vat_base,
        'vat_rate': vat_rate,
        'vat': vat_amount,
        'total': total,
        **payment,
        'raw_text': extracted_text,
    }


def read_pdf_ticket_directory(
    directory: str | Path,
    strict: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Parse all PDFs from a directory.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        Parsed invoice table and parsing-error table.
    """
    pdf_dir = Path(
        directory
    )

    if not pdf_dir.exists():
        raise FileNotFoundError(
            f'PDF directory not found: {pdf_dir}'
        )

    records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    for path in sorted(
        pdf_dir.glob(
            '*.pdf'
        )
    ):
        try:
            records.append(
                read_pdf_ticket(
                    path
                )
            )

        except Exception as exc:
            if strict:
                raise

            errors.append(
                {
                    'source_file': path.name,
                    'error_type': type(exc).__name__,
                    'error_message': str(exc),
                }
            )

    data = pd.DataFrame(
        records
    )

    error_dataframe = pd.DataFrame(
        errors,
        columns=[
            'source_file',
            'error_type',
            'error_message',
        ],
    )

    return (
        data,
        error_dataframe,
    )


# =============================================================================
# DIRECTORY LOADING
# =============================================================================

def _iter_supported_files(
    directory: Path,
    recursive: bool,
) -> list[Path]:
    """Return supported top-level/recursive tabular files."""
    candidates = (
        directory.rglob('*')
        if recursive
        else directory.glob('*')
    )

    return sorted(
        path
        for path in candidates
        if (
            path.is_file()
            and path.suffix.lower()
            in {
                '.xls',
                '.xlsx',
                '.csv',
            }
            and not path.name.startswith(
                '~$'
            )
        )
    )


def _append_dataset(
    grouped_data: dict[str, list[pd.DataFrame]],
    dataset_name: str,
    dataframe: pd.DataFrame,
) -> None:
    """Append a DataFrame to a dataset group if it is non-empty."""
    if dataframe is None:
        return

    if len(dataframe) == 0:
        return

    grouped_data.setdefault(
        dataset_name,
        [],
    ).append(
        dataframe
    )


def load_raw_data_directory(
    data_dir: str | Path,
    recursive: bool = False,
    strict: bool = False,
    excluded_directories: Iterable[str | Path] | None = None,
    verbose: bool = True,
) -> tuple[
    dict[str, pd.DataFrame],
    pd.DataFrame,
]:
    """
    Load supported Excel/CSV files from a raw directory.

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Directory containing raw Excel/CSV data.
    recursive : bool, default=False
        Search recursively when True.
    strict : bool, default=False
        Raise immediately on an unsupported/broken file when True.
    excluded_directories : iterable, optional
        Paths that should be ignored while recursively scanning.
    verbose : bool, default=True
        Print ingestion progress.

    Returns
    -------
    tuple
        Dictionary of concatenated standardized datasets and a DataFrame with
        loading errors.
    """
    directory = Path(
        data_dir
    )

    if not directory.exists():
        raise FileNotFoundError(
            f'Data directory not found: {directory}'
        )

    if not directory.is_dir():
        raise NotADirectoryError(
            f'Expected a directory: {directory}'
        )

    excluded = {
        Path(path).resolve()
        for path in (
            excluded_directories
            or []
        )
    }

    grouped_data: dict[
        str,
        list[pd.DataFrame],
    ] = {}

    errors: list[
        dict[str, str]
    ] = []

    files = _iter_supported_files(
        directory,
        recursive=recursive,
    )

    if not files:
        raise FileNotFoundError(
            f'No supported Excel/CSV files found in: {directory}'
        )

    for file_path in files:
        resolved = file_path.resolve()

        if any(
            parent == excluded_path
            or excluded_path in parent.parents
            for excluded_path in excluded
            for parent in [resolved.parent]
        ):
            continue

        _print_message(
            f'Reading {file_path.name}',
            verbose=verbose,
        )

        try:
            suffix = file_path.suffix.lower()

            if suffix in SUPPORTED_EXCEL_EXTENSIONS:
                dataset_name, data = (
                    read_excel_data_file(
                        file_path
                    )
                )

                _append_dataset(
                    grouped_data,
                    dataset_name,
                    data,
                )

            elif suffix == '.csv':
                csv_datasets = (
                    read_csv_data_file(
                        file_path
                    )
                )

                for dataset_name, data in (
                    csv_datasets.items()
                ):
                    _append_dataset(
                        grouped_data,
                        dataset_name,
                        data,
                    )

        except Exception as exc:
            if strict:
                raise

            errors.append(
                {
                    'source_file': file_path.name,
                    'error_type': type(exc).__name__,
                    'error_message': str(exc),
                }
            )

            _print_message(
                f'Skipped {file_path.name}: {exc}',
                level='WARNING',
                verbose=verbose,
            )

    datasets = {
        dataset_name: pd.concat(
            frames,
            ignore_index=True,
            sort=False,
        )
        for dataset_name, frames
        in grouped_data.items()
    }

    error_dataframe = pd.DataFrame(
        errors,
        columns=[
            'source_file',
            'error_type',
            'error_message',
        ],
    )

    return (
        datasets,
        error_dataframe,
    )


def load_weekly_sales_directory(
    directory: str | Path,
    strict: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all periodic article-sales Excel reports from one folder.
    """
    sales_dir = Path(
        directory
    )

    if not sales_dir.exists():
        raise FileNotFoundError(
            f'Weekly sales directory not found: {sales_dir}'
        )

    frames: list[pd.DataFrame] = []
    errors: list[dict[str, str]] = []

    files = sorted(
        path
        for path in sales_dir.glob('*')
        if (
            path.is_file()
            and path.suffix.lower()
            in SUPPORTED_EXCEL_EXTENSIONS
            and not path.name.startswith(
                '~$'
            )
        )
    )

    for path in files:
        _print_message(
            f'Reading weekly article sales: {path.name}',
            verbose=verbose,
        )

        try:
            raw = _read_excel_raw(
                path
            )

            detected = _detect_file_type(
                raw
            )

            if detected != 'article_sales':
                raise ValueError(
                    'File does not match the expected article-sales format.'
                )

            frames.append(
                read_article_sales(
                    path,
                    raw=raw,
                )
            )

        except Exception as exc:
            if strict:
                raise

            errors.append(
                {
                    'source_file': path.name,
                    'error_type': type(exc).__name__,
                    'error_message': str(exc),
                }
            )

    data = (
        pd.concat(
            frames,
            ignore_index=True,
            sort=False,
        )
        if frames
        else pd.DataFrame()
    )

    error_dataframe = pd.DataFrame(
        errors,
        columns=[
            'source_file',
            'error_type',
            'error_message',
        ],
    )

    return (
        data,
        error_dataframe,
    )


# =============================================================================
# PARQUET-SAFE TYPE STANDARDIZATION
# =============================================================================

def _make_parquet_safe(
    dataframe: pd.DataFrame,
    dataset_name: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Make pandas object columns safe for Arrow/Parquet persistence.

    Pandas object columns can legally mix Python types (for example an
    `entered_by` column containing both integer employee codes and text names),
    while Apache Arrow requires one consistent type per column.

    Bronze should preserve values rather than guess their semantic meaning.
    Therefore mixed/object textual columns are converted to pandas' nullable
    string dtype before they are written to Parquet.
    """
    data = dataframe.copy()

    for column in data.columns:
        if data[column].dtype != 'object':
            continue

        non_null = data[column].dropna()

        if non_null.empty:
            data[column] = data[column].astype('string')
            continue

        python_types = {
            type(value).__name__
            for value in non_null.head(5000)
        }

        data[column] = (
            data[column]
            .astype('string')
        )

        _print_message(
            f'BRONZE | {dataset_name}.{column}: object column converted to '
            f'nullable string for Parquet compatibility '
            f'(observed Python types: {sorted(python_types)}).',
            verbose=verbose,
        )

    return data


# =============================================================================
# BRONZE PERSISTENCE
# =============================================================================

def _ensure_directory(
    directory: str | Path,
) -> Path:
    path = Path(
        directory
    )

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def save_bronze_datasets(
    datasets: dict[str, pd.DataFrame],
    bronze_dir: str | Path,
    verbose: bool = True,
) -> dict[str, Path]:
    """
    Persist standardized ingestion outputs as Bronze Parquet snapshots.
    """
    output_dir = _ensure_directory(
        bronze_dir
    )

    saved_paths: dict[
        str,
        Path,
    ] = {}

    for dataset_name, dataframe in sorted(
        datasets.items()
    ):
        if dataframe is None:
            continue

        filename = BRONZE_FILENAMES.get(
            dataset_name,
            f'{dataset_name}_raw.parquet',
        )

        output_path = (
            output_dir
            / filename
        )

        parquet_dataframe = _make_parquet_safe(
            dataframe=dataframe,
            dataset_name=dataset_name,
            verbose=verbose,
        )

        parquet_dataframe.to_parquet(
            output_path,
            index=False,
        )

        saved_paths[
            dataset_name
        ] = output_path

        _print_message(
            f'BRONZE | {dataset_name}: '
            f'{len(dataframe):,} rows x '
            f'{len(dataframe.columns):,} columns '
            f'-> {output_path}',
            verbose=verbose,
        )

    return saved_paths


def save_ingestion_errors(
    errors: pd.DataFrame,
    bronze_dir: str | Path,
) -> Path | None:
    """Save ingestion errors for traceability."""
    if errors.empty:
        return None

    output_dir = _ensure_directory(
        bronze_dir
    )

    output_path = (
        output_dir
        / 'ingestion_errors.csv'
    )

    errors.to_csv(
        output_path,
        index=False,
    )

    return output_path


def load_bronze_datasets(
    bronze_dir: str | Path,
    verbose: bool = True,
    console_detail: str = 'summary',
) -> dict[str, pd.DataFrame]:
    """
    Load existing Bronze snapshots.

    Useful when the future main program is configured with DO_INGEST=False and
    DO_DEPURATION=True.
    """
    _validate_console_detail(
        console_detail
    )

    detailed_console = (
        verbose
        and console_detail == 'detailed'
    )

    directory = Path(
        bronze_dir
    )

    if not directory.exists():
        raise FileNotFoundError(
            f'Bronze directory not found: {directory}'
        )

    datasets: dict[
        str,
        pd.DataFrame,
    ] = {}

    inverse_names = {
        filename: dataset_name
        for dataset_name, filename
        in BRONZE_FILENAMES.items()
    }

    for path in sorted(
        directory.glob(
            '*.parquet'
        )
    ):
        dataset_name = (
            inverse_names.get(
                path.name
            )
            or re.sub(
                r'_raw$',
                '',
                path.stem,
            )
        )

        datasets[
            dataset_name
        ] = pd.read_parquet(
            path
        )

        _print_message(
            f'Loaded BRONZE | {dataset_name}: '
            f'{len(datasets[dataset_name]):,} rows',
            verbose=detailed_console,
        )

    if not datasets:
        raise FileNotFoundError(
            f'No Bronze parquet files found in: {directory}'
        )

    if (
        verbose
        and console_detail == 'summary'
    ):
        total_rows = sum(
            len(
                dataframe
            )
            for dataframe in datasets.values()
        )
        print(
            '[INFO] Loaded existing Bronze: '
            f'{len(datasets):,} datasets | '
            f'{total_rows:,} total rows.'
        )

    return datasets


# =============================================================================
# USER-STRUCTURED DATA LOADING
# =============================================================================

def load_structured_user_data(
    input_paths: dict[str, str | Path],
    verbose: bool = True,
    console_detail: str = 'summary',
) -> dict[str, pd.DataFrame]:
    """
    Load already structured user datasets when ingestion is intentionally
    bypassed.

    Each dict key becomes the dataset name.

    Example
    -------
    input_paths = {
        'tickets': 'data/my_tickets.parquet',
        'reservas': 'data/my_reservations.csv',
    }
    """
    _validate_console_detail(
        console_detail
    )

    detailed_console = (
        verbose
        and console_detail == 'detailed'
    )

    datasets: dict[
        str,
        pd.DataFrame,
    ] = {}

    for dataset_name, input_path in input_paths.items():
        path = Path(
            input_path
        )

        if not path.exists():
            raise FileNotFoundError(
                f'Input path not found for {dataset_name}: {path}'
            )

        suffix = path.suffix.lower()

        if suffix == '.parquet':
            data = pd.read_parquet(
                path
            )

        elif suffix == '.csv':
            data = _read_csv_flexible(
                path
            )

        elif suffix in SUPPORTED_EXCEL_EXTENSIONS:
            data = pd.read_excel(
                path
            )

            data = _normalize_dataframe_columns(
                data
            )

        else:
            raise ValueError(
                f'Unsupported structured input for {dataset_name}: {path}'
            )

        datasets[
            dataset_name
        ] = data

        _print_message(
            f'Loaded structured input | {dataset_name}: '
            f'{len(data):,} rows x {len(data.columns):,} columns',
            verbose=detailed_console,
        )

    if (
        verbose
        and console_detail == 'summary'
    ):
        total_rows = sum(
            len(
                dataframe
            )
            for dataframe in datasets.values()
        )
        print(
            '[INFO] Loaded structured inputs: '
            f'{len(datasets):,} datasets | '
            f'{total_rows:,} total rows.'
        )

    return datasets



def _validate_console_detail(
    console_detail: str,
) -> None:
    if console_detail not in {
        'summary',
        'detailed',
    }:
        raise ValueError(
            "console_detail must be 'summary' or 'detailed'."
        )


def _print_compact_ingestion_summary(
    datasets: dict[str, pd.DataFrame],
    errors: pd.DataFrame,
    bronze_dir: str | Path,
    verbose: bool = True,
) -> None:
    """
    Print a concise ingestion summary.

    Full data snapshots and ingestion-error details remain persisted exactly as
    before; this helper only reduces terminal noise.
    """
    if not verbose:
        return

    total_rows = sum(
        len(
            dataframe
        )
        for dataframe in datasets.values()
    )
    total_missing = sum(
        int(
            dataframe.isna().sum().sum()
        )
        for dataframe in datasets.values()
    )

    print(
        '[INFO] Ingestion completed: '
        f'{len(datasets):,} Bronze datasets | '
        f'{total_rows:,} total rows | '
        f'{total_missing:,} missing cells.'
    )

    dataset_parts = [
        (
            f'{dataset_name}='
            f'{len(dataframe):,}x{len(dataframe.columns):,}'
        )
        for dataset_name, dataframe
        in sorted(
            datasets.items()
        )
    ]

    print(
        '[INFO] Bronze datasets: '
        + ' | '.join(
            dataset_parts
        )
    )

    if not errors.empty:
        error_path = (
            Path(
                bronze_dir
            )
            / 'ingestion_errors.csv'
        )

        print(
            '[WARNING] '
            f'{len(errors):,} ingestion warning/error record(s). '
            f'Full details: {error_path}'
        )


# =============================================================================
# SUMMARIES
# =============================================================================

def summarize_datasets(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Create a compact summary of the currently loaded datasets."""
    rows = [
        {
            'dataset': dataset_name,
            'rows': len(dataframe),
            'columns': len(dataframe.columns),
            'missing_cells': int(
                dataframe.isna().sum().sum()
            ),
            'memory_mb': round(
                dataframe.memory_usage(
                    deep=True
                ).sum()
                / 1024**2,
                3,
            ),
        }
        for dataset_name, dataframe
        in sorted(
            datasets.items()
        )
    ]

    return pd.DataFrame(
        rows
    )


def print_dataset_summary(
    datasets: dict[str, pd.DataFrame],
) -> None:
    """Print a human-readable ingestion summary."""
    summary = summarize_datasets(
        datasets
    )

    if summary.empty:
        print(
            'No datasets loaded.'
        )
        return

    print(
        '\nINGESTION SUMMARY'
    )
    print(
        summary.to_string(
            index=False
        )
    )


# =============================================================================
# MASTER INGESTION FUNCTION
# =============================================================================

def run_ingestion(
    paths: IngestionPaths,
    recursive_raw_scan: bool = False,
    strict: bool = False,
    verbose: bool = True,
    console_detail: str = 'summary',
) -> dict[str, pd.DataFrame]:
    """
    Master ingestion entry point.

    This is the function that the future `analysis_prediction.py` main file
    should normally call.

    Workflow
    --------
    1. Read supported Excel/CSV sources from `raw_dir`.
    2. Read periodic article-sales reports from `weekly_sales_dir`.
    3. Read ticket/invoice PDFs from `pdf_tickets_dir`.
    4. Concatenate equal dataset families.
    5. Save one standardized Parquet snapshot per dataset in Bronze.
    6. Save ingestion errors separately.
    7. Return the in-memory standardized datasets.

    Console
    -------
    `console_detail='summary'` is the default and prints only a compact
    ingestion result. `console_detail='detailed'` restores file-by-file,
    Parquet-conversion and persistence diagnostics.

    Notes
    -----
    This function performs TECHNICAL STANDARDIZATION, not semantic depuration.
    Missing-value treatment, outlier decisions, consistency rules and Silver
    creation belong in `ap_depuration.py`.
    """
    _validate_console_detail(
        console_detail
    )

    detailed_console = (
        verbose
        and console_detail == 'detailed'
    )

    _print_header(
        'INGESTION -> BRONZE',
        verbose=verbose,
    )

    raw_dir = Path(
        paths.raw_dir
    )

    bronze_dir = _ensure_directory(
        paths.bronze_dir
    )

    excluded_directories = [
        path
        for path in [
            paths.weekly_sales_dir,
            paths.pdf_tickets_dir,
            bronze_dir,
        ]
        if path is not None
    ]

    datasets, errors = (
        load_raw_data_directory(
            data_dir=raw_dir,
            recursive=recursive_raw_scan,
            strict=strict,
            excluded_directories=excluded_directories,
            verbose=detailed_console,
        )
    )

    if (
        paths.weekly_sales_dir is not None
        and Path(
            paths.weekly_sales_dir
        ).exists()
    ):
        weekly_sales, weekly_errors = (
            load_weekly_sales_directory(
                directory=paths.weekly_sales_dir,
                strict=strict,
                verbose=detailed_console,
            )
        )

        if not weekly_sales.empty:
            if 'ventas' in datasets:
                datasets['ventas'] = pd.concat(
                    [
                        datasets['ventas'],
                        weekly_sales,
                    ],
                    ignore_index=True,
                    sort=False,
                )
            else:
                datasets['ventas'] = (
                    weekly_sales
                )

        if not weekly_errors.empty:
            errors = pd.concat(
                [
                    errors,
                    weekly_errors,
                ],
                ignore_index=True,
            )

    if (
        paths.pdf_tickets_dir is not None
        and Path(
            paths.pdf_tickets_dir
        ).exists()
    ):
        _print_message(
            f'Reading PDF tickets from: {paths.pdf_tickets_dir}',
            verbose=detailed_console,
        )

        facturas, pdf_errors = (
            read_pdf_ticket_directory(
                directory=paths.pdf_tickets_dir,
                strict=strict,
            )
        )

        if not facturas.empty:
            datasets['facturas'] = (
                facturas
            )

        if not pdf_errors.empty:
            errors = pd.concat(
                [
                    errors,
                    pdf_errors,
                ],
                ignore_index=True,
            )

    if not datasets:
        raise RuntimeError(
            'No supported datasets were successfully ingested.'
        )

    save_bronze_datasets(
        datasets=datasets,
        bronze_dir=bronze_dir,
        verbose=detailed_console,
    )

    error_path = save_ingestion_errors(
        errors=errors,
        bronze_dir=bronze_dir,
    )

    if error_path is not None:
        _print_message(
            f'{len(errors):,} ingestion errors/warnings saved to {error_path}',
            level='WARNING',
            verbose=detailed_console,
        )

    if console_detail == 'detailed':
        print_dataset_summary(
            datasets
        )

        _print_message(
            f'Ingestion completed: {len(datasets)} Bronze datasets available.',
            verbose=verbose,
        )

    else:
        _print_compact_ingestion_summary(
            datasets=datasets,
            errors=errors,
            bronze_dir=bronze_dir,
            verbose=verbose,
        )

    return datasets
