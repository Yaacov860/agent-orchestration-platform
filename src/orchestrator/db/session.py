from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)


def create_engine(database_url: str) -> AsyncEngine:
    # pool_pre_ping: test a pooled connection before use, so a Postgres restart
    # costs one reconnect instead of one failed request.
    return create_async_engine(database_url, pool_pre_ping=True)


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    # expire_on_commit=False: keep loaded values after commit. The default would reload
    # them lazily on next access, which async SQLAlchemy cannot do implicitly.
    return async_sessionmaker(engine, expire_on_commit=False)
