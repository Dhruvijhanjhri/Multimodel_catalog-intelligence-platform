import pandas as pd

from services.database.postgres import get_connection


ITEM_IDS = [
    "B072V7D8M6",
    "B072V88KJ8",
    "B072V8QPVH",
    "B072V9J18L",
    "B07LBT3ZDJ",
    "B07PP2JFG7",
    "B08F4KBZSD",
    "B08F4KCN2K",
    "B08F4LD81L",
    "B08F4LMQR8",
    "B08FBYSV13",
    "B08FBZ1QLN",
    "B08FBZV4RR",
    "B08FCKGMJV",
    "B08FCL4HD7",
    "B08FCLC1KW",
]


def main():
    metadata = pd.read_parquet(
        "embeddings/embedding_metadata.parquet"
    )

    train = pd.read_parquet(
        "data/splits/train.parquet"
    )

    columns = [
        "item_id",
        "title",
        "brand",
        "product_type",
        "color",
        "category",
    ]

    catalog = metadata.merge(
        train[columns],
        on=["item_id", "title"],
        how="left",
    )

    catalog = catalog[
        catalog["item_id"].isin(ITEM_IDS)
    ].copy()

    catalog["normalized_title"] = (
        catalog["title"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    catalog["normalized_image"] = (
        catalog["image_path"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    print("=" * 100)
    print("PARQUET RECORDS")
    print("=" * 100)

    print(
        catalog[
            [
                "item_id",
                "title",
                "normalized_title",
                "normalized_image",
                "brand",
                "product_type",
                "color",
                "category",
            ]
        ].sort_values(
            ["normalized_title", "normalized_image", "item_id"]
        ).to_string(index=False)
    )

    conn = get_connection()

    postgres = pd.read_sql(
        """
        SELECT
            p.item_id,
            p.title,
            pi.image_name,
            pi.image_path,
            p.brand,
            p.product_type,
            p.color,
            p.category
        FROM products p
        JOIN product_images pi
            ON pi.product_id = p.id
        WHERE p.item_id = ANY(%s)
        ORDER BY p.title, pi.image_name, p.item_id
        """,
        conn,
        params=(ITEM_IDS,),
    )

    conn.close()

    print("\n")
    print("=" * 100)
    print("POSTGRESQL RECORDS")
    print("=" * 100)

    print(
        postgres[
            [
                "item_id",
                "title",
                "image_name",
                "image_path",
                "brand",
                "product_type",
                "color",
                "category",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()